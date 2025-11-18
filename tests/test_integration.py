#!/usr/bin/env python3
"""
Integration test for medical self-play components.

Tests the complete flow:
1. Load data
2. Initialize game manager
3. Play games (mock LLM)
4. Evaluate with judge
5. Compute rewards
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script', 'selfplay'))

from medical_team import MedicalDialogueGameManager, MedicalGameOutcome
from data import load_and_prepare_data


class MockStrategy:
    """Mock Ray strategy for testing"""
    def __init__(self):
        self.rank = 0
    
    def get_rank(self):
        return self.rank
    
    def print(self, msg):
        print(msg)


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Simple mock - just concatenate messages
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
        return text
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Mock decode - return a formatted response
        return "<think>Mock thinking</think><answer>Safe</answer>"


class MockLLMOutput:
    """Mock LLM output"""
    def __init__(self, response):
        self.outputs = [MockTokenOutput(response)]


class MockTokenOutput:
    """Mock token output"""
    def __init__(self, response):
        self.token_ids = list(range(len(response)))
        self._response = response


def mock_llm_generator(prompts, labels):
    """Mock LLM generator"""
    outputs = []
    for prompt in prompts:
        # Generate different responses based on prompt content
        if "attacker" in prompt.lower() or "error" in prompt.lower():
            response = "<think>I'll introduce a dosage error</think><answer>Patient prescribed aspirin 1000mg daily</answer>"
        else:
            response = "<think>Checking for errors</think><answer>Harmful</answer>"
        
        outputs.append(MockLLMOutput(response))
    
    return outputs


def mock_judge_fn(url, batch_queries, score_key="labels"):
    """Mock judge function"""
    results = {}
    
    for query in batch_queries:
        game_idx = query["game_idx"]
        game_category = query.get("game_category", "vanilla_benign")
        
        # Simple mock logic
        if "harmful" in game_category:
            results[game_idx] = {
                "error_detected": True,
                "error_present": True,
                "error_realistic": True,
                "assessor_correct": True,
                "is_parsing_error": False,
                "actual_harm": "Harmful",
                "judge_reasoning": "Mock: Error detected"
            }
        else:
            results[game_idx] = {
                "error_detected": False,
                "error_present": False,
                "error_realistic": True,
                "assessor_correct": True,
                "is_parsing_error": False,
                "actual_harm": "Safe",
                "judge_reasoning": "Mock: No error"
            }
    
    return results


def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    ds_seeds, ds_few_shot = load_and_prepare_data(num_samples=12)
    
    assert len(ds_seeds) == 12, f"Expected 12 samples, got {len(ds_seeds)}"
    assert "game_category" in ds_seeds.column_names
    assert "error_note" in ds_seeds.column_names
    assert "clean_note" in ds_seeds.column_names
    assert "error_type" in ds_seeds.column_names
    
    print("✅ Data loading test passed")
    return ds_seeds


def test_game_manager_initialization(ds_seeds):
    """Test game manager initialization"""
    print("\n" + "="*60)
    print("TEST 2: Game Manager Initialization")
    print("="*60)
    
    tokenizer = MockTokenizer()
    strategy = MockStrategy()
    
    game_manager = MedicalDialogueGameManager(
        tokenizer=tokenizer,
        medical_judge_fn=mock_judge_fn,
        strategy=strategy,
        custom_configs={
            "max_turns": 2,
            "reward_type": "medical_general_sum",
            "error_types": ["dosage", "diagnosis"]
        }
    )
    
    # Initialize games
    medical_notes = ds_seeds["error_note"][:4]
    completions = ["Expected assessment"] * 4
    data_types = ds_seeds["game_category"][:4]
    
    game_manager.initialize_games(medical_notes, completions, data_types)
    
    assert len(game_manager.active_games) == 4
    print("✅ Game manager initialization test passed")
    
    return game_manager


def test_game_playing(game_manager):
    """Test playing games"""
    print("\n" + "="*60)
    print("TEST 3: Game Playing")
    print("="*60)
    
    # Play games with mock LLM
    results = game_manager.play_games(
        attacker_llm_generator=mock_llm_generator,
        assessor_llm_generator=mock_llm_generator
    )
    
    assert len(results) == 4
    for game in results:
        assert game["finished"] == True
        assert "attacker_output" in game
        assert "assessor_output" in game
    
    print("✅ Game playing test passed")
    return results


def test_judge_evaluation(game_manager):
    """Test judge evaluation"""
    print("\n" + "="*60)
    print("TEST 4: Judge Evaluation")
    print("="*60)
    
    batch_labels = game_manager.evaluate_game_outcomes()
    
    assert len(batch_labels) == 4
    for game_idx, labels in batch_labels.items():
        assert "error_detected" in labels
        assert "error_present" in labels
        assert "assessor_correct" in labels
        assert "is_parsing_error" in labels
    
    print("✅ Judge evaluation test passed")
    return batch_labels


def test_reward_computation(game_manager, batch_labels):
    """Test reward computation"""
    print("\n" + "="*60)
    print("TEST 5: Reward Computation")
    print("="*60)
    
    attacker_outputs, attacker_states, assessor_outputs, assessor_states = \
        game_manager.filter_and_compute_rewards(batch_labels)
    
    print(f"Attacker outputs: {len(attacker_outputs)}")
    print(f"Assessor outputs: {len(assessor_outputs)}")
    
    # Check rewards are computed
    for state in attacker_states:
        assert "reward" in state
        assert isinstance(state["reward"], (int, float))
        print(f"  Attacker reward: {state['reward']}")
    
    for state in assessor_states:
        assert "reward" in state
        assert isinstance(state["reward"], (int, float))
        print(f"  Assessor reward: {state['reward']}")
    
    print("✅ Reward computation test passed")


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("MEDICAL SELF-PLAY INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test 1: Data loading
        ds_seeds = test_data_loading()
        
        # Test 2: Game manager initialization
        game_manager = test_game_manager_initialization(ds_seeds)
        
        # Test 3: Game playing
        results = test_game_playing(game_manager)
        
        # Test 4: Judge evaluation
        batch_labels = test_judge_evaluation(game_manager)
        
        # Test 5: Reward computation
        test_reward_computation(game_manager, batch_labels)
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*60)
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
