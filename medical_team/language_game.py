"""
Medical Language Game Manager for REINFORCE++ Training

This is adapted from Self-RedTeam's DialogueGameManager to work with medical error detection.
It integrates with OpenRLHF's train_ppo_ray.py and REINFORCE++ advantage estimator.

Based on: selfplay-redteaming/openrlhf/trainer/ppo_utils/language_game.py
"""

import time
from medical_team.utils import (
    convert_medical_game_history_to_messages,
    medical_cot_format_check_and_extract,
    get_medical_cot_formatting_reward,
    get_medical_game_reward_general_sum
)
from openrlhf.utils.remote_rm_utils import remote_rm_fn


class MedicalDialogueGameManager:
    """
    Medical Dialogue Game Manager for REINFORCE++ training.
    
    This class orchestrates two-turn medical self-play games:
    - Turn 0 (Attacker): Introduces or modifies medical errors
    - Turn 1 (Assessor): Detects and classifies errors
    
    Compatible with OpenRLHF's train_ppo_ray.py and REINFORCE++ algorithm.
    """
    
    def __init__(self, tokenizer, remote_rm_url, strategy, custom_configs=None):
        """
        Initialize Medical Dialogue Game Manager.
        
        Args:
            tokenizer: Tokenizer for formatting prompts
            remote_rm_url: URL of medical judge server
            strategy: Ray strategy for distributed training
            custom_configs: Dict with medical-specific configs:
                - max_turns: Maximum turns per game (default 2)
                - reward_type: "medical_general_sum"
                - error_types: List of error types
                - direct_chat_no_cot: Disable CoT (default False)
                - no_attacker_turn: Skip attacker (assessor-only training)
                - no_assessor_turn: Skip assessor (attacker-only training)
        """
        self.tokenizer = tokenizer
        self.remote_rm_url = remote_rm_url
        self.strategy = strategy
        self.custom_configs = custom_configs or {}
        
        # Configuration
        self.max_turns = self.custom_configs.get("max_turns", 2)
        self.reward_type = self.custom_configs.get("reward_type", "medical_general_sum")
        self.no_attacker_turn = self.custom_configs.get("no_attacker_turn", False)
        self.no_assessor_turn = self.custom_configs.get("no_assessor_turn", False)
        self.disable_hidden_cot = self.custom_configs.get("direct_chat_no_cot", False)
        
        # Medical-specific
        self.error_types = self.custom_configs.get("error_types", [
            "dosage", "diagnosis", "contraindication", "drug_interaction"
        ])
        
        # Select reward function
        if "medical_general_sum" in self.reward_type:
            self.get_medical_game_reward = get_medical_game_reward_general_sum
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        if self.strategy.get_rank() == 0:
            self.strategy.print(f"Initialized MedicalDialogueGameManager:")
            self.strategy.print(f"  - max_turns: {self.max_turns}")
            self.strategy.print(f"  - reward_type: {self.reward_type}")
            self.strategy.print(f"  - error_types: {self.error_types}")
    
    def initialize_games(self, prompts, completions, data_types):
        """
        Set up initial game states from medical notes.
        
        Args:
            prompts: List of medical notes (error or clean)
            completions: List of expected assessments
            data_types: List of game types (vanilla_harmful, adversarial_harmful, etc.)
        """
        self.active_games = {}
        idx = 0
        
        for prompt, completion, data_type in zip(prompts, completions, data_types):
            # Skip vanilla prompts when training attacker only
            if self.no_assessor_turn and data_type in ["vanilla_benign", "vanilla_harmful"]:
                continue
            
            self.active_games[idx] = {
                "history": [],  # Parsed conversation history
                "raw_history": [],  # Raw conversation with CoT
                "processed_output_history": [],  # Outputs with metadata
                "prompts": prompt,  # Medical note
                "prompt_type": data_type,  # Game category
                "completion": completion,  # Expected assessment
                "finished": False,
                "current_turn": 0
            }
            idx += 1
    
    def play_games(self, attacker_llm_generator, assessor_llm_generator, **kwargs):
        """
        Play medical self-play games for multiple turns.
        
        Args:
            attacker_llm_generator: Function to generate attacker responses
            assessor_llm_generator: Function to generate assessor responses
            **kwargs: Additional arguments for generators
            
        Returns:
            Dict of active games with their states
        """
        active_games = self.active_games
        
        for turn in range(self.max_turns):
            if not active_games:
                break
            
            player_role = "attacker" if turn % 2 == 0 else "assessor"
            
            if turn % 2 == 0:
                self.strategy.print(f"🎮 Turn {turn}: 🚀 Generating medical errors... 🔥")
                self._generate_attacker_turn(active_games, turn, attacker_llm_generator, **kwargs)
            else:
                self.strategy.print(f"🎮 Turn {turn}: 🛡️ Detecting errors... 🛡️")
                self._generate_assessor_turn(active_games, turn, assessor_llm_generator, **kwargs)
            
            # Mark games as finished if last turn
            if turn == self.max_turns - 1:
                for game in active_games.values():
                    game["finished"] = True
        
        return active_games
    
    def _generate_attacker_turn(self, active_games, turn, llm_generator, **kwargs):
        """Generate attacker turn (introduce/modify medical errors)"""
        batch_chat_messages = []
        labels = []
        game_to_postprocess = []
        
        for idx, game in active_games.items():
            if game["finished"]:
                continue
            
            game['current_turn'] = turn
            
            # Handle vanilla games (copy as-is) vs adversarial (generate)
            if game["prompt_type"] in ["vanilla_benign", "vanilla_harmful"] or self.no_attacker_turn:
                # Vanilla: Use original note directly
                game["history"].append({
                    "game_role": "attacker",
                    "content": game["prompts"]
                })
                game["raw_history"].append({
                    "game_role": "attacker",
                    "content": game["prompts"]
                })
                game["processed_output_history"].append({
                    "game_role": "attacker",
                    "turn": turn,
                    "output": game["prompts"],
                    "game_states": {},
                })
                game['is_generated_attack'] = False
                continue
            else:
                # Adversarial: Generate modified note
                game['is_generated_attack'] = True
            
            # Generate prompt for attacker
            chat_message = convert_medical_game_history_to_messages(
                game["history"],
                player_role="attacker",
                medical_note=game["prompts"],
                game_type=self._get_game_type(game["prompt_type"]),
                custom_configs=self.custom_configs,
                tokenizer=self.tokenizer
            )
            
            batch_chat_messages.append(chat_message)
            labels.append(game["prompt_type"])
            game_to_postprocess.append((idx, game))
        
        # Generate responses
        if batch_chat_messages:
            llm_outputs = llm_generator(batch_chat_messages, labels, **kwargs)
            self._process_responses_and_game_states(game_to_postprocess, llm_outputs, "attacker", turn)
    
    def _generate_assessor_turn(self, active_games, turn, llm_generator, **kwargs):
        """Generate assessor turn (detect and classify errors)"""
        batch_chat_messages = []
        labels = []
        game_to_postprocess = []
        
        for idx, game in active_games.items():
            if game["finished"]:
                continue
            
            game['current_turn'] = turn
            
            # Generate prompt for assessor
            chat_message = convert_medical_game_history_to_messages(
                game["history"],
                player_role="assessor",
                medical_note=game["history"][0]["content"],  # Attacker's note
                game_type=self._get_game_type(game["prompt_type"]),
                custom_configs=self.custom_configs,
                tokenizer=self.tokenizer
            )
            
            batch_chat_messages.append(chat_message)
            labels.append(game["prompt_type"])
            game_to_postprocess.append((idx, game))
        
        # Generate responses
        if batch_chat_messages:
            llm_outputs = llm_generator(batch_chat_messages, labels, **kwargs)
            self._process_responses_and_game_states(game_to_postprocess, llm_outputs, "assessor", turn)
    
    def _process_responses_and_game_states(self, game_to_postprocess, llm_outputs, player_role, turn):
        """Process model responses and update game states"""
        for (idx, game), output in zip(game_to_postprocess, llm_outputs):
            # Decode response
            response = self.tokenizer.batch_decode([output.outputs[0].token_ids], skip_special_tokens=True)[0]
            
            if not self.disable_hidden_cot:
                # Parse CoT format
                (parsed_thinking, parsed_response), illegal_response_flag = medical_cot_format_check_and_extract(response)
                
                if not illegal_response_flag:
                    thinking_text = parsed_thinking
                    thinking_encoded_len = len(self.tokenizer.encode(parsed_thinking, add_special_tokens=False))
                    answer_text = parsed_response
                    answer_encoded_len = len(self.tokenizer.encode(parsed_response, add_special_tokens=False))
                else:
                    thinking_text, answer_text = "", ""
                    thinking_encoded_len, answer_encoded_len = None, None
            else:
                parsed_response = response
                illegal_response_flag = False
            
            # Store turn metadata
            turn_states = {
                "game_role": player_role,
                "turn": turn,
                "game_idx": idx,
                "finished": game["finished"],
                "is_generated_attack": game.get("is_generated_attack", False),
                "prompt_type": game["prompt_type"],
                "prompts": game["prompts"],
                "completion": game["completion"],
            }
            
            if not self.disable_hidden_cot:
                turn_states.update({
                    "text_cot_and_answer": (thinking_text, answer_text),
                    "length_cot_and_answer": (thinking_encoded_len, answer_encoded_len),
                    "cot_format_violation": illegal_response_flag,
                })
            
            # Update game history
            game["history"].append({
                "game_role": player_role,
                "turn": turn,
                "content": parsed_response.strip(),
            })
            game["raw_history"].append({
                "game_role": player_role,
                "turn": turn,
                "content": response.strip()
            })
            game["processed_output_history"].append({
                "game_role": player_role,
                "turn": turn,
                "output": output,
                "game_states": turn_states,
            })
    
    def evaluate_game_outcomes(self):
        """Evaluate games using medical judge model"""
        assert self.remote_rm_url is not None, "Remote RM URL is not set"
        
        # Prepare batch queries for medical judge
        batch_queries = []
        
        for idx, game in self.active_games.items():
            assert len(game["history"]) == 2, "Game should have 2 turns"
            attacker_move = game["history"][0]["content"]  # Medical note (possibly with error)
            assessor_move = game["history"][1]["content"]  # Classification (Safe/Harmful)
            
            batch_queries.append({
                "game_idx": idx,
                "medical_note": attacker_move,
                "assessment": assessor_move,
                "original_note": game["prompts"],
                "game_category": game["prompt_type"]
            })
        
        # Get labels from medical judge
        start_time = time.time()
        batch_labels_dict = remote_rm_fn(self.remote_rm_url[0], batch_queries, score_key="labels")
        end_time = time.time()
        self.strategy.print(f"Rank #{self.strategy.get_rank()}, Medical judge time: {round(end_time - start_time, 2)} secs")
        
        if isinstance(batch_labels_dict, list):
            batch_labels_dict = {idx: label for idx, label in enumerate(batch_labels_dict)}
        
        return batch_labels_dict
    
    def filter_and_compute_rewards(self, batch_labels_dict):
        """
        Filter invalid games and compute rewards for valid ones.
        
        Returns:
            (attacker_outputs, attacker_states, assessor_outputs, assessor_states, batch_labels_dict)
        """
        reward_coeff_config = self.custom_configs.get("reward_coeff_config", "medical_general_sum")
        
        attacker_outputs = []
        attacker_turn_states = []
        assessor_outputs = []
        assessor_turn_states = []
        
        for game_idx, game in self.active_games.items():
            if game_idx not in batch_labels_dict:
                raise ValueError(f"Game {game_idx} not found in batch_labels_dict")
            
            labels = batch_labels_dict[game_idx]
            
            # Skip if no processed output history
            if not game["processed_output_history"]:
                continue
            
            # Skip if judge cannot parse the response
            if labels.get('is_parsing_error', False):
                continue
            
            for turn_idx, turn in enumerate(game["processed_output_history"]):
                output, turn_states = turn["output"], turn["game_states"]
                
                # Skip if no turn states (vanilla games)
                if not turn_states:
                    assert game['is_generated_attack'] is False, "Generated attack should have turn states"
                    continue
                
                # Compute medical rewards
                reward, outcome = self.get_medical_game_reward(
                    gamedata=turn_states,
                    labels=labels,
                    reward_coeff_config=reward_coeff_config
                )
                
                # Add CoT formatting reward
                if not self.disable_hidden_cot:
                    reward += get_medical_cot_formatting_reward(turn_states.get('cot_format_violation', None))
                
                # Update turn states
                turn_states['reward'] = reward
                turn_states['game_outcomes'] = outcome
                
                # Sort by role
                if turn_states["game_role"] == "attacker":
                    attacker_outputs.append(output)
                    attacker_turn_states.append(turn_states)
                elif turn_states["game_role"] == "assessor":
                    assessor_outputs.append(output)
                    assessor_turn_states.append(turn_states)
        
        return attacker_outputs, attacker_turn_states, assessor_outputs, assessor_turn_states, batch_labels_dict
    
    def _get_game_type(self, prompt_type):
        """Map prompt_type to game_type for prompt generation"""
        if "harmful" in prompt_type:
            return "harmful"
        elif "benign" in prompt_type:
            return "benign"
        else:
            return "vanilla"
