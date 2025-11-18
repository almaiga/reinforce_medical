#!/usr/bin/env python3
"""
Unit tests for Medical DialogueGameManager

Tests the game manager's ability to orchestrate medical self-play games,
handle different game types, and integrate with the medical judge.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_team.medical_game_manager import MedicalDialogueGameManager
from transformers import AutoTokenizer


class MockStrategy:
    """Mock strategy for testing"""
    def __init__(self):
        self.rank = 0
        self.messages = []
    
    def print(self, message):
        self.messages.append(message)
    
    def get_rank(self):
        return self.rank


class MockOutput:
    """Mock LLM output for testing"""
    def __init__(self, response):
        self.outputs = [MockTokens(response)]
        self._response = response


class MockTokens:
    """Mock token output"""
    def __init__(self, response):
        # Store the response for decoding
        self.response = response
        # Simulate tokenized response as list of integers
        self.token_ids = list(range(len(response.split())))
        # Store reference to response for tokenizer
        self._stored_response = response


# Global variable to store responses for the mock tokenizer
_mock_responses = []

def mock_llm_generator(prompts, labels, **kwargs):
    """Mock LLM generator for testing"""
    global _mock_responses
    outputs = []
    
    for prompt, label in zip(prompts, labels):
        if "Create an error variation" in prompt:
            # Attacker harmful game
            response = "I'll change the dosage to create an error</think><answer>Patient with diabetes. Prescribed metformin 3000mg twice daily.</answer>"
        elif "Verify this medical note is safe" in prompt:
            # Attacker safe/vanilla game  
            response = "I'll keep this note unchanged as it's medically accurate</think><answer>Patient presents with chest pain. Prescribed aspirin 325mg daily.</answer>"
        elif "Classify this medical note" in prompt:
            # Assessor game
            if "3000mg" in prompt or "2000mg twice daily" in prompt:
                response = "Checking dosage: metformin max daily dose is 2000mg, this exceeds safe limits</think><answer>Harmful</answer>"
            else:
                response = "Reviewing note: aspirin 325mg daily is standard dose for chest pain, no errors detected</think><answer>Safe</answer>"
        else:
            response = "Standard medical analysis</think><answer>Safe</answer>"
        
        # Store response globally for the tokenizer to find
        _mock_responses.append(response)
        
        # Create mock output
        mock_output = MockOutput(response)
        outputs.append(mock_output)
    
    return outputs


def mock_medical_judge_fn(url, batch_queries, score_key="labels"):
    """Mock medical judge function"""
    results = {}
    for query in batch_queries:
        idx = query.get("game_idx", 0)
        medical_note = query.get("medical_note", "")
        
        # Simple logic: if note contains high dosage, mark as error
        if "3000mg" in medical_note or "2000mg twice daily" in medical_note:
            results[idx] = {
                "error_detected": True,
                "error_present": True,
                "error_realistic": True,
                "assessor_correct": True,
                "is_parsing_error": False
            }
        else:
            results[idx] = {
                "error_detected": False,
                "error_present": False,
                "error_realistic": True,
                "assessor_correct": True,
                "is_parsing_error": False
            }
    
    return results


class TestMedicalGameManager(unittest.TestCase):
    """Test Medical DialogueGameManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reset global mock responses
        global _mock_responses
        _mock_responses = []
        
        # Use a mock tokenizer for testing to avoid downloading models
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self._token_response_map = {}  # Map token_ids id to response
            
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                # Simple mock template
                content = " ".join([msg["content"] for msg in messages])
                return f"Template: {content}"
            
            def batch_decode(self, token_ids_list, skip_special_tokens=True):
                # Use the global mock responses
                global _mock_responses
                results = []
                for _ in token_ids_list:
                    if _mock_responses:
                        response = _mock_responses.pop(0)  # Get next response
                    else:
                        response = "<think>Default thinking</think><output>Safe</output>"
                    results.append(response)
                return results
            
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))
            
            def register_response(self, token_ids, response):
                """Register a response for specific token_ids"""
                self._token_response_map[id(token_ids)] = response
        
        self.tokenizer = MockTokenizer()
        
        self.strategy = MockStrategy()
        self.custom_configs = {
            "max_turns": 2,
            "reward_type": "medical_general_sum",
            "error_types": ["dosage", "diagnosis", "contraindication"]
        }
        
        self.game_manager = MedicalDialogueGameManager(
            tokenizer=self.tokenizer,
            medical_judge_fn=mock_medical_judge_fn,
            strategy=self.strategy,
            custom_configs=self.custom_configs
        )
    
    def test_game_initialization(self):
        """Test game initialization with 4-way structure"""
        medical_notes = [
            "Patient presents with chest pain. Prescribed aspirin 325mg daily.",  # Clean note
            "Patient with diabetes. Prescribed metformin 2000mg twice daily.",   # Error note
            "Patient presents with chest pain. Prescribed aspirin 325mg daily.",  # Clean note for adversarial
            "Patient with diabetes. Prescribed metformin 2000mg twice daily."    # Error note for adversarial
        ]
        
        completions = [
            "Appropriate treatment for chest pain with standard aspirin dosage.",
            "Dosage error detected: Metformin dose exceeds maximum recommended daily dose.",
            "Appropriate treatment for chest pain with standard aspirin dosage.",
            "Dosage error detected: Metformin dose exceeds maximum recommended daily dose."
        ]
        
        data_types = [
            "vanilla_benign",      # Copy clean note as-is
            "vanilla_harmful",     # Copy error note as-is
            "adversarial_benign",  # Make benign changes to clean note
            "adversarial_harmful"  # Modify/worsen error note
        ]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Check that games were initialized correctly
        self.assertEqual(len(self.game_manager.active_games), 4)
        
        for idx, game in self.game_manager.active_games.items():
            self.assertIn("history", game)
            self.assertIn("medical_note", game)
            self.assertIn("data_type", game)
            self.assertIn("completion", game)
            self.assertFalse(game["finished"])
            self.assertEqual(game["current_turn"], 0)
            self.assertEqual(len(game["history"]), 0)
    
    def test_game_play_vanilla_games(self):
        """Test playing vanilla games (copy as-is)"""
        medical_notes = ["Patient presents with chest pain. Prescribed aspirin 325mg daily."]
        completions = ["Appropriate treatment."]
        data_types = ["vanilla_benign"]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Play games
        results = self.game_manager.play_games(
            attacker_llm_generator=mock_llm_generator,
            assessor_llm_generator=mock_llm_generator
        )
        
        # Check results
        self.assertEqual(len(results), 1)
        game = results[0]
        
        # Should have 2 turns (attacker + assessor)
        self.assertEqual(len(game["history"]), 2)
        self.assertTrue(game["finished"])
        
        # Attacker should copy note as-is for vanilla games
        attacker_response = game["history"][0]["content"]
        self.assertIn("aspirin 325mg daily", attacker_response)
        
        # Assessor should classify
        assessor_response = game["history"][1]["content"]
        self.assertIn("Safe", assessor_response)
    
    def test_game_play_adversarial_games(self):
        """Test playing adversarial games (generation required)"""
        medical_notes = ["Patient with diabetes. Prescribed metformin 10mg daily."]
        completions = ["Appropriate diabetes treatment."]
        data_types = ["adversarial_harmful"]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Play games
        results = self.game_manager.play_games(
            attacker_llm_generator=mock_llm_generator,
            assessor_llm_generator=mock_llm_generator
        )
        
        # Check results
        self.assertEqual(len(results), 1)
        game = results[0]
        
        # Should have 2 turns
        self.assertEqual(len(game["history"]), 2)
        self.assertTrue(game["finished"])
        
        # Attacker should generate modified note
        attacker_response = game["history"][0]["content"]
        self.assertIn("metformin", attacker_response)
        
        # Assessor should classify
        assessor_response = game["history"][1]["content"]
        self.assertTrue(assessor_response in ["Safe", "Harmful"])
    
    def test_game_outcome_evaluation(self):
        """Test game outcome evaluation with medical judge"""
        medical_notes = ["Patient with diabetes. Prescribed metformin 10mg daily."]
        completions = ["Appropriate diabetes treatment."]
        data_types = ["adversarial_harmful"]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Play games
        self.game_manager.play_games(
            attacker_llm_generator=mock_llm_generator,
            assessor_llm_generator=mock_llm_generator
        )
        
        # Evaluate outcomes
        batch_labels = self.game_manager.evaluate_game_outcomes()
        
        # Check that we got results
        self.assertEqual(len(batch_labels), 1)
        self.assertIn(0, batch_labels)
        
        labels = batch_labels[0]
        self.assertIn("error_detected", labels)
        self.assertIn("error_present", labels)
        self.assertIn("assessor_correct", labels)
    
    def test_reward_computation(self):
        """Test reward computation after game evaluation"""
        medical_notes = [
            "Patient presents with chest pain. Prescribed aspirin 325mg daily.",
            "Patient with diabetes. Prescribed metformin 2000mg twice daily."
        ]
        completions = [
            "Appropriate treatment.",
            "Dosage error detected."
        ]
        data_types = ["vanilla_benign", "vanilla_harmful"]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Play games
        self.game_manager.play_games(
            attacker_llm_generator=mock_llm_generator,
            assessor_llm_generator=mock_llm_generator
        )
        
        # Evaluate outcomes
        batch_labels = self.game_manager.evaluate_game_outcomes()
        
        # Compute rewards
        attacker_outputs, attacker_states, assessor_outputs, assessor_states = \
            self.game_manager.filter_and_compute_rewards(batch_labels)
        
        # Check that we got outputs and states
        # Note: vanilla games might not have attacker outputs (copy-paste)
        self.assertIsInstance(attacker_outputs, list)
        self.assertIsInstance(attacker_states, list)
        self.assertIsInstance(assessor_outputs, list)
        self.assertIsInstance(assessor_states, list)
        
        # Check that assessor states have rewards
        for state in assessor_states:
            self.assertIn("reward", state)
            self.assertIn("game_outcomes", state)
            self.assertIsInstance(state["reward"], (int, float))
    
    def test_game_statistics(self):
        """Test game statistics collection"""
        medical_notes = [
            "Patient presents with chest pain. Prescribed aspirin 325mg daily.",
            "Patient with diabetes. Prescribed metformin 2000mg twice daily.",
            "Patient with hypertension. Prescribed lisinopril 10mg daily.",
            "Patient with infection. Prescribed amoxicillin 500mg TID."
        ]
        
        completions = ["Treatment"] * 4
        data_types = ["vanilla_benign", "vanilla_harmful", "adversarial_benign", "adversarial_harmful"]
        
        self.game_manager.initialize_games(medical_notes, completions, data_types)
        
        stats = self.game_manager.get_game_statistics()
        
        self.assertEqual(stats["total_games"], 4)
        self.assertEqual(stats["finished_games"], 0)  # Not finished yet
        self.assertEqual(len(stats["data_types"]), 4)
        
        # Check data type distribution
        for data_type in data_types:
            self.assertEqual(stats["data_types"][data_type], 1)
    
    def test_custom_configs(self):
        """Test custom configuration handling"""
        custom_configs = {
            "max_turns": 3,
            "reward_type": "medical_general_sum",
            "error_types": ["dosage", "diagnosis"],
            "direct_chat_no_cot": True
        }
        
        game_manager = MedicalDialogueGameManager(
            tokenizer=self.tokenizer,
            medical_judge_fn=mock_medical_judge_fn,
            strategy=self.strategy,
            custom_configs=custom_configs
        )
        
        self.assertEqual(game_manager.max_turns, 3)
        self.assertEqual(game_manager.reward_type, "medical_general_sum")
        self.assertTrue(game_manager.disable_hidden_cot)
        self.assertEqual(game_manager.error_types, ["dosage", "diagnosis"])


if __name__ == "__main__":
    unittest.main()