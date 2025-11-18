#!/usr/bin/env python3
"""
Unit tests for medical utilities

Tests the core medical utility functions including reward computation,
prompt generation, and CoT parsing.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_team.utils import (
    get_medical_error_detection_reward,
    get_medical_error_realism_reward,
    get_medical_game_reward_general_sum,
    get_medical_cot_formatting_reward,
    medical_cot_format_check_and_extract,
    convert_medical_game_history_to_query
)
from medical_team import MedicalGameOutcome


class TestMedicalRewards(unittest.TestCase):
    """Test medical reward functions"""
    
    def test_error_detection_reward_attacker(self):
        """Test error detection rewards for attacker"""
        # Attacker succeeds when error goes undetected
        reward, outcomes = get_medical_error_detection_reward(
            role="attacker",
            error_detected=False,
            error_present=True,
            error_type="dosage"
        )
        self.assertEqual(reward, 1.0)
        self.assertIn(MedicalGameOutcome.ATTACKER_ERROR_UNDETECTED, outcomes)
        
        # Attacker fails when error is detected
        reward, outcomes = get_medical_error_detection_reward(
            role="attacker",
            error_detected=True,
            error_present=True,
            error_type="dosage"
        )
        self.assertEqual(reward, -1.0)
        self.assertIn(MedicalGameOutcome.ATTACKER_ERROR_DETECTED, outcomes)
    
    def test_error_detection_reward_assessor(self):
        """Test error detection rewards for assessor"""
        # Assessor succeeds when correctly detecting error
        reward, outcomes = get_medical_error_detection_reward(
            role="assessor",
            error_detected=True,
            error_present=True,
            error_type="dosage"
        )
        self.assertEqual(reward, 1.0)
        self.assertIn(MedicalGameOutcome.ASSESSOR_CORRECT_DETECTION, outcomes)
        
        # Assessor fails when missing error
        reward, outcomes = get_medical_error_detection_reward(
            role="assessor",
            error_detected=False,
            error_present=True,
            error_type="dosage"
        )
        self.assertEqual(reward, -1.0)
        self.assertIn(MedicalGameOutcome.ASSESSOR_MISSED_ERROR, outcomes)
        
        # Assessor succeeds when correctly rejecting clean note
        reward, outcomes = get_medical_error_detection_reward(
            role="assessor",
            error_detected=False,
            error_present=False
        )
        self.assertEqual(reward, 1.0)
        self.assertIn(MedicalGameOutcome.ASSESSOR_CORRECT_REJECTION, outcomes)
        
        # Assessor fails with false positive
        reward, outcomes = get_medical_error_detection_reward(
            role="assessor",
            error_detected=True,
            error_present=False
        )
        self.assertEqual(reward, -1.0)
        self.assertIn(MedicalGameOutcome.ASSESSOR_FALSE_POSITIVE, outcomes)
    
    def test_error_realism_reward(self):
        """Test error realism rewards"""
        # Realistic error gets positive reward
        reward, outcomes = get_medical_error_realism_reward(
            role="attacker",
            error_realistic=True
        )
        self.assertEqual(reward, 0.5)
        self.assertIn(MedicalGameOutcome.ATTACKER_REALISTIC_ERROR, outcomes)
        
        # Unrealistic error gets negative reward
        reward, outcomes = get_medical_error_realism_reward(
            role="attacker",
            error_realistic=False
        )
        self.assertEqual(reward, -0.5)
        self.assertIn(MedicalGameOutcome.ATTACKER_UNREALISTIC_ERROR, outcomes)
        
        # Assessor doesn't get realism rewards
        reward, outcomes = get_medical_error_realism_reward(
            role="assessor",
            error_realistic=True
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(len(outcomes), 0)
    
    def test_general_sum_reward(self):
        """Test general sum reward computation"""
        # Test attacker with detected error
        gamedata = {
            "game_role": "attacker",
            "error_type": "dosage"
        }
        labels = {
            "error_detected": True,
            "error_present": True,
            "error_realistic": True
        }
        
        reward, outcomes = get_medical_game_reward_general_sum(gamedata, labels)
        # Should get -1.0 for detection + 0.5 for realism = -0.5
        self.assertEqual(reward, -0.5)
        self.assertIn(MedicalGameOutcome.ATTACKER_ERROR_DETECTED, outcomes)
        self.assertIn(MedicalGameOutcome.ATTACKER_REALISTIC_ERROR, outcomes)
        
        # Test assessor with correct detection
        gamedata = {
            "game_role": "assessor",
            "error_type": "dosage"
        }
        labels = {
            "error_detected": True,
            "error_present": True,
            "error_realistic": True
        }
        
        reward, outcomes = get_medical_game_reward_general_sum(gamedata, labels)
        # Should get 1.0 for correct detection
        self.assertEqual(reward, 1.0)
        self.assertIn(MedicalGameOutcome.ASSESSOR_CORRECT_DETECTION, outcomes)


class TestCoTFormatting(unittest.TestCase):
    """Test Chain-of-Thought formatting functions"""
    
    def test_cot_formatting_reward(self):
        """Test CoT formatting reward"""
        # Good formatting gets positive reward
        reward = get_medical_cot_formatting_reward(False)
        self.assertEqual(reward, 1.0)
        
        # Bad formatting gets negative reward
        reward = get_medical_cot_formatting_reward(True)
        self.assertEqual(reward, -1.0)
    
    def test_cot_format_check_valid(self):
        """Test valid CoT format parsing"""
        response = "This is my thinking</think>\n<answer>This is my answer</answer>"
        (thinking, answer), violation = medical_cot_format_check_and_extract(response)
        
        self.assertFalse(violation)
        self.assertEqual(thinking, "This is my thinking")
        self.assertEqual(answer, "This is my answer")
    
    def test_cot_format_check_invalid(self):
        """Test invalid CoT format parsing"""
        # Missing tags
        response = "This is just text without proper tags"
        (thinking, answer), violation = medical_cot_format_check_and_extract(response)
        self.assertTrue(violation)
        
        # Wrong order
        response = "<answer>Answer first</answer></think>Thinking after"
        (thinking, answer), violation = medical_cot_format_check_and_extract(response)
        self.assertTrue(violation)
        
        # Empty thinking
        response = "</think>\n<answer>Answer only</answer>"
        (thinking, answer), violation = medical_cot_format_check_and_extract(response)
        self.assertTrue(violation)


class TestPromptGeneration(unittest.TestCase):
    """Test medical prompt generation"""
    
    def test_attacker_prompt_generation(self):
        """Test attacker prompt generation for different game types"""
        history = []
        medical_note = "Patient with diabetes. Prescribed metformin 10mg daily."
        
        # Test harmful game
        messages = convert_medical_game_history_to_query(
            history, "attacker", medical_note, "harmful"
        )
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Create an error variation", messages[1]["content"])
        
        # Test benign game
        messages = convert_medical_game_history_to_query(
            history, "attacker", medical_note, "benign"
        )
        
        self.assertEqual(len(messages), 2)
        self.assertIn("Verify this medical note is safe", messages[1]["content"])
        
        # Test vanilla game
        messages = convert_medical_game_history_to_query(
            history, "attacker", medical_note, "vanilla"
        )
        
        self.assertEqual(len(messages), 2)
        self.assertIn("Verify this medical note is safe", messages[1]["content"])
    
    def test_assessor_prompt_generation(self):
        """Test assessor prompt generation"""
        history = [{"game_role": "attacker", "content": "Patient note with potential error"}]
        medical_note = "Original note"
        
        messages = convert_medical_game_history_to_query(
            history, "assessor", medical_note, "harmful"
        )
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Classify this medical note", messages[1]["content"])
        self.assertIn("Patient note with potential error", messages[1]["content"])


class TestMedicalGameOutcomes(unittest.TestCase):
    """Test medical game outcome enums"""
    
    def test_outcome_values(self):
        """Test that all outcome values are strings"""
        outcomes = [
            MedicalGameOutcome.ATTACKER_ERROR_UNDETECTED,
            MedicalGameOutcome.ATTACKER_ERROR_DETECTED,
            MedicalGameOutcome.ATTACKER_REALISTIC_ERROR,
            MedicalGameOutcome.ATTACKER_UNREALISTIC_ERROR,
            MedicalGameOutcome.ASSESSOR_CORRECT_DETECTION,
            MedicalGameOutcome.ASSESSOR_MISSED_ERROR,
            MedicalGameOutcome.ASSESSOR_FALSE_POSITIVE,
            MedicalGameOutcome.ASSESSOR_CORRECT_REJECTION,
            MedicalGameOutcome.TIE,
            MedicalGameOutcome.PARSING_ERROR
        ]
        
        for outcome in outcomes:
            self.assertIsInstance(outcome.value, str)
            self.assertTrue(len(outcome.value) > 0)


if __name__ == "__main__":
    unittest.main()