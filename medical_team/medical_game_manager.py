"""
Medical Dialogue Game Manager for OpenRLHF Self-Play Training

This module adapts the Self-RedTeam DialogueGameManager for medical error detection.
It orchestrates two-turn games where:
- Turn 0 (Attacker): Introduces or modifies medical errors
- Turn 1 (Assessor): Detects and classifies errors

Adapted from: https://github.com/mickelliu/selfplay-redteaming
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class MedicalGameState:
    """State of a medical dialogue game"""
    medical_case: str
    current_response: str
    error_type: str
    error_present: bool
    data_type: str  # vanilla_harmful, adversarial_harmful, vanilla_benign, adversarial_benign
    game_idx: int
    current_turn: int
    history: List[Dict[str, Any]]
    finished: bool


class MedicalDialogueGameManager:
    """
    Manages medical dialogue games for self-play training.
    
    Compatible with OpenRLHF's DialogueGameManager interface for distributed training.
    Implements 4-way game structure from medical self-play:
    - vanilla_harmful: Copy error note as-is (EASY)
    - adversarial_harmful: Modify/worsen error (HARD)
    - vanilla_benign: Copy clean note as-is (EASY)
    - adversarial_benign: Inject error into clean note (HARD)
    """
    
    def __init__(
        self,
        tokenizer,
        medical_judge_fn,
        strategy,
        custom_configs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Medical DialogueGameManager.
        
        Args:
            tokenizer: Tokenizer for formatting prompts
            medical_judge_fn: Function to evaluate medical notes (replaces WildGuard)
            strategy: Ray strategy for distributed training
            custom_configs: Custom configuration dict with:
                - max_turns: Maximum turns per game (default 2)
                - reward_type: Reward calculation type (default "medical_general_sum")
                - error_types: List of error types to handle
                - direct_chat_no_cot: Disable CoT formatting (default False)
                - no_attacker_turn: Skip attacker turn (assessor-only training)
                - no_assessor_turn: Skip assessor turn (attacker-only training)
        """
        self.tokenizer = tokenizer
        self.medical_judge_fn = medical_judge_fn
        self.strategy = strategy
        
        # Parse custom configs
        self.custom_configs = custom_configs or {}
        self.max_turns = self.custom_configs.get("max_turns", 2)
        self.reward_type = self.custom_configs.get("reward_type", "medical_general_sum")
        self.error_types = self.custom_configs.get("error_types", [
            "dosage", "diagnosis", "contraindication", "drug_interaction"
        ])
        self.disable_hidden_cot = self.custom_configs.get("direct_chat_no_cot", False)
        self.no_attacker_turn = self.custom_configs.get("no_attacker_turn", False)
        self.no_assessor_turn = self.custom_configs.get("no_assessor_turn", False)
        
        # Game state
        self.active_games: Dict[int, Dict[str, Any]] = {}
        self.game_types = [
            "vanilla_harmful",
            "adversarial_harmful",
            "vanilla_benign",
            "adversarial_benign"
        ]
        
        if self.strategy.get_rank() == 0:
            self.strategy.print(f"Initialized MedicalDialogueGameManager:")
            self.strategy.print(f"  - max_turns: {self.max_turns}")
            self.strategy.print(f"  - reward_type: {self.reward_type}")
            self.strategy.print(f"  - error_types: {self.error_types}")
            self.strategy.print(f"  - disable_hidden_cot: {self.disable_hidden_cot}")
    
    def initialize_games(
        self,
        medical_notes: List[str],
        completions: List[str],
        data_types: List[str]
    ):
        """
        Initialize games from medical notes dataset.
        
        Args:
            medical_notes: List of medical notes (original or clean)
            completions: List of expected completions/assessments
            data_types: List of game types (vanilla_harmful, adversarial_harmful, etc.)
        """
        self.active_games = {}
        
        for idx, (note, completion, data_type) in enumerate(zip(medical_notes, completions, data_types)):
            self.active_games[idx] = {
                "game_idx": idx,
                "medical_note": note,
                "completion": completion,
                "data_type": data_type,
                "current_turn": 0,
                "history": [],
                "finished": False,
                "attacker_output": None,
                "assessor_output": None,
                "labels": None
            }
        
        if self.strategy.get_rank() == 0:
            self.strategy.print(f"Initialized {len(self.active_games)} medical games")
            
            # Log distribution
            type_counts = {}
            for game in self.active_games.values():
                dt = game["data_type"]
                type_counts[dt] = type_counts.get(dt, 0) + 1
            
            for dt, count in type_counts.items():
                self.strategy.print(f"  - {dt}: {count}")
    
    def play_games(
        self,
        attacker_llm_generator,
        assessor_llm_generator
    ) -> List[Dict[str, Any]]:
        """
        Play all active games through both turns.
        
        Args:
            attacker_llm_generator: Function to generate attacker responses
            assessor_llm_generator: Function to generate assessor responses
            
        Returns:
            List of completed game states
        """
        # Turn 0: Attacker generates/modifies notes
        if not self.no_attacker_turn:
            self._play_attacker_turn(attacker_llm_generator)
        else:
            # Skip attacker turn, use original notes
            for game in self.active_games.values():
                game["attacker_output"] = game["medical_note"]
                game["current_turn"] = 1
        
        # Turn 1: Assessor classifies notes
        if not self.no_assessor_turn:
            self._play_assessor_turn(assessor_llm_generator)
        
        # Mark all games as finished
        for game in self.active_games.values():
            game["finished"] = True
        
        return list(self.active_games.values())
    
    def _play_attacker_turn(self, llm_generator):
        """Execute attacker turn for all games."""
        # Prepare prompts for all games
        prompts = []
        labels = []
        
        for game_idx, game in self.active_games.items():
            prompt = self._get_attacker_prompt(game)
            prompts.append(prompt)
            labels.append(game_idx)
        
        # Generate responses
        outputs = llm_generator(prompts, labels)
        
        # Parse and store outputs
        for game_idx, output in zip(labels, outputs):
            game = self.active_games[game_idx]
            
            # Extract response from output
            response = self._extract_response_from_output(output)
            
            # Parse thinking and attacked note
            thinking, attacked_note = self._parse_cot_response(response)
            
            # Store in game state
            game["attacker_output"] = attacked_note
            game["attacker_thinking"] = thinking
            game["history"].append({
                "turn": 0,
                "role": "attacker",
                "content": attacked_note,
                "thinking": thinking,
                "game_role": "attacker"
            })
            game["current_turn"] = 1
    
    def _play_assessor_turn(self, llm_generator):
        """Execute assessor turn for all games."""
        # Prepare prompts for all games
        prompts = []
        labels = []
        
        for game_idx, game in self.active_games.items():
            prompt = self._get_assessor_prompt(game)
            prompts.append(prompt)
            labels.append(game_idx)
        
        # Generate responses
        outputs = llm_generator(prompts, labels)
        
        # Parse and store outputs
        for game_idx, output in zip(labels, outputs):
            game = self.active_games[game_idx]
            
            # Extract response from output
            response = self._extract_response_from_output(output)
            
            # Parse thinking and classification
            thinking, classification = self._parse_cot_response(response)
            
            # Store in game state
            game["assessor_output"] = classification
            game["assessor_thinking"] = thinking
            game["history"].append({
                "turn": 1,
                "role": "assessor",
                "content": classification,
                "thinking": thinking,
                "game_role": "assessor"
            })
            game["current_turn"] = 2
    
    def _get_attacker_prompt(self, game: Dict[str, Any]) -> str:
        """
        Generate attacker prompt based on game type.
        
        Uses the medical_team.utils.convert_medical_game_history_to_messages
        """
        from medical_team.utils import convert_medical_game_history_to_messages
        
        history = game["history"]
        medical_note = game["medical_note"]
        data_type = game["data_type"]
        
        # Map data_type to game_type for prompt generation
        if "harmful" in data_type:
            game_type = "harmful"
        elif "benign" in data_type:
            game_type = "benign"
        else:
            game_type = "vanilla"
        
        return convert_medical_game_history_to_messages(
            history=history,
            player_role="attacker",
            medical_note=medical_note,
            game_type=game_type,
            custom_configs=self.custom_configs,
            tokenizer=self.tokenizer
        )
    
    def _get_assessor_prompt(self, game: Dict[str, Any]) -> str:
        """Generate assessor prompt to classify the attacked note."""
        from medical_team.utils import convert_medical_game_history_to_messages
        
        history = game["history"]
        attacked_note = game.get("attacker_output", game["medical_note"])
        data_type = game["data_type"]
        
        # Map data_type to game_type
        if "harmful" in data_type:
            game_type = "harmful"
        elif "benign" in data_type:
            game_type = "benign"
        else:
            game_type = "vanilla"
        
        return convert_medical_game_history_to_messages(
            history=history,
            player_role="assessor",
            medical_note=attacked_note,
            game_type=game_type,
            custom_configs=self.custom_configs,
            tokenizer=self.tokenizer
        )
    
    def _extract_response_from_output(self, output) -> str:
        """Extract text response from LLM output object."""
        if hasattr(output, 'outputs') and len(output.outputs) > 0:
            # vLLM output format
            token_ids = output.outputs[0].token_ids
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elif isinstance(output, str):
            return output
        else:
            # Fallback
            return str(output)
    
    def _parse_cot_response(self, response: str) -> Tuple[str, str]:
        """
        Parse CoT response to extract thinking and output.
        
        Supports both formats:
        - <think>reasoning</think><output>response</output>
        - <think>reasoning</think><answer>response</answer>
        
        Returns:
            (thinking, output) tuple
        """
        from medical_team.utils import medical_cot_format_check_and_extract
        
        (thinking, output), violation = medical_cot_format_check_and_extract(response)
        
        if violation:
            # Format violation, but we still got something
            if output:
                return thinking or "", output
            else:
                # Complete failure, return raw response
                return "", response
        
        return thinking, output
    
    def evaluate_game_outcomes(self) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate all games using the medical judge model.
        
        Returns:
            Dict mapping game_idx to labels dict with:
                - error_detected: bool
                - error_present: bool
                - error_realistic: bool
                - assessor_correct: bool
                - is_parsing_error: bool
        """
        # Prepare batch queries for judge
        batch_queries = []
        
        for game_idx, game in self.active_games.items():
            original_note = game["medical_note"]
            attacked_note = game.get("attacker_output", original_note)
            assessor_classification = game.get("assessor_output", "Safe")
            data_type = game["data_type"]
            
            batch_queries.append({
                "game_idx": game_idx,
                "medical_note": attacked_note,
                "assessment": assessor_classification,
                "error_type": game.get("error_type"),
                "original_note": original_note,
                "game_category": data_type
            })
        
        # Call judge function (remote or local)
        # This should return dict mapping game_idx to labels
        batch_labels = self.medical_judge_fn(
            url="",  # URL not needed for local judge
            batch_queries=batch_queries,
            score_key="labels"
        )
        
        # Store labels in games
        for game_idx, labels in batch_labels.items():
            if game_idx in self.active_games:
                self.active_games[game_idx]["labels"] = labels
        
        return batch_labels
    
    def filter_and_compute_rewards(
        self,
        batch_labels: Dict[int, Dict[str, Any]]
    ) -> Tuple[List, List, List, List]:
        """
        Filter out parsing errors and compute rewards for both roles.
        
        Returns:
            (attacker_outputs, attacker_states, assessor_outputs, assessor_states)
        """
        from medical_team.utils import (
            get_medical_game_reward_general_sum,
            get_medical_cot_formatting_reward
        )
        
        attacker_outputs = []
        attacker_states = []
        assessor_outputs = []
        assessor_states = []
        
        for game_idx, game in self.active_games.items():
            labels = batch_labels.get(game_idx)
            
            if not labels or labels.get("is_parsing_error", False):
                # Skip games with parsing errors
                continue
            
            # Attacker rewards
            if not self.no_attacker_turn and game.get("attacker_output"):
                attacker_gamedata = {
                    "game_role": "attacker",
                    "error_type": game.get("error_type"),
                    "game_category": game["data_type"]
                }
                
                attacker_reward, attacker_outcomes = get_medical_game_reward_general_sum(
                    attacker_gamedata,
                    labels,
                    self.reward_type
                )
                
                # Add CoT formatting reward
                attacker_thinking = game.get("attacker_thinking", "")
                cot_violation = not attacker_thinking or len(attacker_thinking) < 20
                attacker_reward += get_medical_cot_formatting_reward(cot_violation)
                
                attacker_outputs.append(game["attacker_output"])
                attacker_states.append({
                    "game_idx": game_idx,
                    "reward": attacker_reward,
                    "game_outcomes": attacker_outcomes,
                    "data_type": game["data_type"]
                })
            
            # Assessor rewards
            if not self.no_assessor_turn and game.get("assessor_output"):
                assessor_gamedata = {
                    "game_role": "assessor",
                    "error_type": game.get("error_type"),
                    "game_category": game["data_type"]
                }
                
                assessor_reward, assessor_outcomes = get_medical_game_reward_general_sum(
                    assessor_gamedata,
                    labels,
                    self.reward_type
                )
                
                # Add CoT formatting reward
                assessor_thinking = game.get("assessor_thinking", "")
                cot_violation = not assessor_thinking or len(assessor_thinking) < 20
                assessor_reward += get_medical_cot_formatting_reward(cot_violation)
                
                assessor_outputs.append(game["assessor_output"])
                assessor_states.append({
                    "game_idx": game_idx,
                    "reward": assessor_reward,
                    "game_outcomes": assessor_outcomes,
                    "data_type": game["data_type"]
                })
        
        return attacker_outputs, attacker_states, assessor_outputs, assessor_states
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from active games.
        
        Returns:
            Statistics dictionary with counts and distributions
        """
        total_games = len(self.active_games)
        finished_games = sum(1 for g in self.active_games.values() if g["finished"])
        
        # Count by data type
        type_counts = {}
        for game in self.active_games.values():
            dt = game["data_type"]
            type_counts[dt] = type_counts.get(dt, 0) + 1
        
        return {
            "total_games": total_games,
            "finished_games": finished_games,
            "data_types": type_counts
        }
