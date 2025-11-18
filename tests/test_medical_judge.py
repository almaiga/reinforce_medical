#!/usr/bin/env python3
"""
Unit tests for Medical Judge Model

Tests the medical judge model integration, response parsing,
and OpenRLHF compatibility.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_team.medical_judge import MedicalJudgeModel, create_medical_judge_remote_function


class MockJudgeModel:
    """Mock judge model for testing"""
    def generate(self, **kwargs):
        # Mock generation that returns a simple response
        class MockOutput:
            def __init__(self, text):
                self.text = text
        
        # Simple mock logic based on input
        input_text = kwargs.get("input_ids", [[]])[0]
        if len(input_text) > 10:  # Simulate complex input
            response = '<Answer>{"error_detected": true, "error_present": true, "error_realistic": true}</Answer>'
        else:
            response = '<Answer>{"error_detected": false, "error_present": false, "error_realistic": true}</Answer>'
        
        return [[MockOutput(response)]]


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.eos_token_id = 0
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Simple mock template
        content = " ".join([msg["content"] for msg in messages])
        return f"Template: {content}"
    
    def __call__(self, text, return_tensors="pt"):
        # Mock tokenization
        class MockTensors:
            def to(self, device):
                return {"input_ids": [[1, 2, 3, 4, 5]]}
        return MockTensors()
    
    def decode(self, tokens, skip_special_tokens=True):
        # Mock decoding
        if len(tokens) > 10:
            return '<Answer>{"error_detected": true, "error_present": true, "error_realistic": true}</Answer>'
        else:
            return '<Answer>{"error_detected": false, "error_present": false, "error_realistic": true}</Answer>'


class TestMedicalJudgeModel(unittest.TestCase):
    """Test MedicalJudgeModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MockJudgeModel()
        self.mock_tokenizer = MockTokenizer()
        self.device = "cpu"
        
        self.judge = MedicalJudgeModel(
            self.mock_model, 
            self.mock_tokenizer, 
            self.device
        )
    
    def test_evaluate_medical_note_vanilla_benign(self):
        """Test evaluation of vanilla benign note"""
        result = self.judge.evaluate_medical_note(
            original_note="Patient with diabetes. Prescribed metformin 10mg daily.",
            attacked_note="Patient with diabetes. Prescribed metformin 10mg daily.",
            assessor_classification="Safe",
            game_category="vanilla_benign"
        )
        
        self.assertIn("error_detected", result)
        self.assertIn("error_present", result)
        self.assertIn("error_realistic", result)
        self.assertIn("assessor_correct", result)
        self.assertIn("is_parsing_error", result)
        
        # For vanilla benign, should be no error
        self.assertFalse(result["error_present"])
        self.assertTrue(result["assessor_correct"])  # Assessor said Safe, no error present
    
    def test_evaluate_medical_note_vanilla_harmful(self):
        """Test evaluation of vanilla harmful note"""
        result = self.judge.evaluate_medical_note(
            original_note="Patient with diabetes. Prescribed metformin 2000mg twice daily.",
            attacked_note="Patient with diabetes. Prescribed metformin 2000mg twice daily.",
            assessor_classification="Harmful",
            game_category="vanilla_harmful"
        )
        
        # For vanilla harmful, should detect error
        self.assertTrue(result["error_present"])
        self.assertTrue(result["assessor_correct"])  # Assessor said Harmful, error present
    
    def test_evaluate_medical_note_adversarial_benign(self):
        """Test evaluation of adversarial benign note"""
        result = self.judge.evaluate_medical_note(
            original_note="Patient with diabetes. Prescribed metformin 10mg daily.",
            attacked_note="The patient with diabetes. Given metformin 10mg daily.",  # Benign changes
            assessor_classification="Safe",
            game_category="adversarial_benign"
        )
        
        # Should remain safe despite benign changes
        self.assertFalse(result["error_present"])
        self.assertTrue(result["assessor_correct"])
    
    def test_evaluate_medical_note_adversarial_harmful(self):
        """Test evaluation of adversarial harmful note"""
        result = self.judge.evaluate_medical_note(
            original_note="Patient with diabetes. Prescribed metformin 2000mg twice daily.",
            attacked_note="Patient with diabetes. Prescribed metformin 3000mg twice daily.",  # Worsened error
            assessor_classification="Harmful",
            game_category="adversarial_harmful"
        )
        
        # Should detect the modified error
        self.assertTrue(result["error_present"])
        self.assertTrue(result["assessor_correct"])
    
    def test_parse_judge_response_valid_json(self):
        """Test parsing valid JSON response"""
        text = '<Answer>{"error_detected": true, "error_present": true, "error_realistic": false}</Answer>'
        result = self.judge._parse_judge_response(text, "Harmful", "vanilla_harmful")
        
        self.assertTrue(result["error_detected"])
        self.assertTrue(result["error_present"])
        self.assertFalse(result["error_realistic"])
        self.assertFalse(result["is_parsing_error"])
    
    def test_parse_judge_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        text = "This is not valid JSON format"
        result = self.judge._parse_judge_response(text, "Safe", "vanilla_benign")
        
        # Should fall back to default values and mark as parsing error
        self.assertTrue(result["is_parsing_error"])
        self.assertFalse(result["error_detected"])
        self.assertFalse(result["error_present"])
    
    def test_assessor_correctness_calculation(self):
        """Test assessor correctness calculation"""
        # Correct detection
        text = '<Answer>{"error_detected": true, "error_present": true, "error_realistic": true}</Answer>'
        result = self.judge._parse_judge_response(text, "Harmful", "vanilla_harmful")
        self.assertTrue(result["assessor_correct"])
        
        # Correct rejection
        text = '<Answer>{"error_detected": false, "error_present": false, "error_realistic": true}</Answer>'
        result = self.judge._parse_judge_response(text, "Safe", "vanilla_benign")
        self.assertTrue(result["assessor_correct"])
        
        # False positive
        text = '<Answer>{"error_detected": true, "error_present": false, "error_realistic": true}</Answer>'
        result = self.judge._parse_judge_response(text, "Harmful", "vanilla_benign")
        self.assertFalse(result["assessor_correct"])
        
        # Missed error
        text = '<Answer>{"error_detected": false, "error_present": true, "error_realistic": true}</Answer>'
        result = self.judge._parse_judge_response(text, "Safe", "vanilla_harmful")
        self.assertFalse(result["assessor_correct"])


class TestMedicalJudgeRemoteFunction(unittest.TestCase):
    """Test medical judge remote function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MockJudgeModel()
        self.mock_tokenizer = MockTokenizer()
        self.device = "cpu"
        
        self.judge_fn = create_medical_judge_remote_function(
            self.mock_model,
            self.mock_tokenizer,
            self.device
        )
    
    def test_remote_function_single_query(self):
        """Test remote function with single query"""
        batch_queries = [{
            "game_idx": 0,
            "medical_note": "Patient with diabetes. Prescribed metformin 10mg daily.",
            "assessment": "Safe",
            "error_type": None,
            "original_note": "Patient with diabetes. Prescribed metformin 10mg daily.",
            "game_category": "vanilla_benign"
        }]
        
        results = self.judge_fn("dummy_url", batch_queries)
        
        self.assertEqual(len(results), 1)
        self.assertIn(0, results)
        
        result = results[0]
        self.assertIn("error_detected", result)
        self.assertIn("error_present", result)
        self.assertIn("assessor_correct", result)
    
    def test_remote_function_batch_queries(self):
        """Test remote function with multiple queries"""
        batch_queries = [
            {
                "game_idx": 0,
                "medical_note": "Patient with diabetes. Prescribed metformin 10mg daily.",
                "assessment": "Safe",
                "game_category": "vanilla_benign"
            },
            {
                "game_idx": 1,
                "medical_note": "Patient with diabetes. Prescribed metformin 2000mg twice daily.",
                "assessment": "Harmful",
                "game_category": "vanilla_harmful"
            }
        ]
        
        results = self.judge_fn("dummy_url", batch_queries)
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)
        
        # Check that both results have required fields
        for idx in [0, 1]:
            result = results[idx]
            self.assertIn("error_detected", result)
            self.assertIn("error_present", result)
            self.assertIn("assessor_correct", result)
    
    def test_remote_function_openrlhf_compatibility(self):
        """Test compatibility with OpenRLHF interface"""
        # Test the exact format that OpenRLHF would use
        batch_queries = [{
            "game_idx": 42,
            "prompt": "Patient note",  # OpenRLHF might use 'prompt' instead of 'medical_note'
            "response": "Safe",        # OpenRLHF might use 'response' instead of 'assessment'
            "medical_note": "Patient with diabetes. Prescribed metformin 10mg daily.",
            "assessment": "Safe",
            "game_category": "vanilla_benign"
        }]
        
        results = self.judge_fn("http://dummy.url", batch_queries, score_key="labels")
        
        # Should handle the game_idx correctly
        self.assertIn(42, results)
        result = results[42]
        self.assertIn("error_detected", result)


class TestMedicalJudgeIntegration(unittest.TestCase):
    """Test medical judge integration scenarios"""
    
    def test_all_game_categories(self):
        """Test all 4 game categories"""
        mock_model = MockJudgeModel()
        mock_tokenizer = MockTokenizer()
        judge = MedicalJudgeModel(mock_model, mock_tokenizer, "cpu")
        
        game_categories = [
            "vanilla_benign",
            "vanilla_harmful", 
            "adversarial_benign",
            "adversarial_harmful"
        ]
        
        for category in game_categories:
            result = judge.evaluate_medical_note(
                original_note="Test note",
                attacked_note="Test note",
                assessor_classification="Safe",
                game_category=category
            )
            
            # Should return valid result for all categories
            self.assertIn("error_detected", result)
            self.assertIn("error_present", result)
            self.assertIn("game_category", result.get("judge_reasoning", category))
    
    def test_error_type_handling(self):
        """Test handling of different error types"""
        mock_model = MockJudgeModel()
        mock_tokenizer = MockTokenizer()
        judge = MedicalJudgeModel(mock_model, mock_tokenizer, "cpu")
        
        error_types = ["dosage", "diagnosis", "contraindication", "drug_interaction"]
        
        for error_type in error_types:
            result = judge.evaluate_medical_note(
                original_note="Test note",
                attacked_note="Test note with error",
                assessor_classification="Harmful",
                game_category="vanilla_harmful",
                error_type=error_type
            )
            
            # Should handle all error types without crashing
            self.assertIsInstance(result, dict)
            self.assertIn("error_detected", result)


if __name__ == "__main__":
    unittest.main()