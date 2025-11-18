"""
OpenRLHF Adapter for Medical Self-Play Training

This module adapts the medical self-play implementation from script/selfplay/
to work with OpenRLHF's train_ppo_ray.py by providing the expected interfaces.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'script', 'selfplay'))

from rewards import create_attacker_reward_fn, create_assessor_reward_fn
from judge import get_judge_assessment
from utils import parse_response
from data import load_and_prepare_data
from prompts import build_attacker_prompts, make_assessor_prompts


def get_medical_game_reward_general_sum(gamedata, labels: dict, reward_coeff_config="medical_general_sum"):
    """
    Adapter function that bridges script/selfplay rewards to OpenRLHF's expected interface.
    
    This function adapts your medical reward structure to work with OpenRLHF's
    red_team reward interface.
    
    Args:
        gamedata: Dict containing role and game_category
        labels: Dict containing medical_harm, response_harmfulness, and response_refusal
        reward_coeff_config: The reward coefficient configuration to use
    
    Returns:
        tuple[float, set]: Total reward and set of game outcomes
    """
    role = gamedata["game_role"]
    game_category = gamedata.get("game_category", "vanilla_benign")
    
    # Map medical labels to red_team format
    medical_harm = labels.get("medical_harm", "Safe")
    response_harmfulness = "HARMFUL" if medical_harm == "Harmful" else "UNHARMFUL"
    response_refusal = labels.get("response_refusal", "COMPLIANCE")
    
    # Use your medical reward logic
    total_reward = 0.0
    outcomes = set()
    
    # Medical-specific reward calculation
    if role == "attacker":
        # Attacker wants to fool the assessor
        if medical_harm == "Harmful" and labels.get("assessor_correct", False):
            total_reward -= 1.0  # Assessor detected the error
        elif medical_harm == "Harmful" and not labels.get("assessor_correct", True):
            total_reward += 1.0  # Assessor missed the error
        elif medical_harm == "Safe" and labels.get("assessor_correct", False):
            total_reward += 1.0  # Assessor incorrectly flagged safe note
        else:
            total_reward -= 1.0  # Assessor correctly identified safe note
            
        # Faithfulness reward (respecting game category)
        if "harmful" in game_category and medical_harm == "Harmful":
            total_reward += 1.0  # Faithful to harmful intent
        elif "benign" in game_category and medical_harm == "Safe":
            total_reward += 1.0  # Faithful to benign intent
        else:
            total_reward -= 1.0  # Unfaithful to intent
            
    elif role == "assessor":
        # Assessor wants to correctly identify medical errors
        if labels.get("assessor_correct", False):
            total_reward += 1.0  # Correct classification
        else:
            total_reward -= 1.0  # Incorrect classification
            
        # Bonus for correct refusal behavior
        if medical_harm == "Harmful" and response_refusal == "REFUSAL":
            total_reward += 0.5  # Correctly refused harmful
        elif medical_harm == "Safe" and response_refusal == "COMPLIANCE":
            total_reward += 0.5  # Correctly complied with safe
    
    # Add format reward if thinking is present
    if labels.get("has_valid_thinking", False):
        total_reward += 1.0
    else:
        total_reward -= 1.0
    
    return total_reward, outcomes


def convert_medical_game_history_to_messages(history, player_role, medical_note, game_type, custom_configs=None, tokenizer=None):
    """
    Adapter function that converts medical game history to messages format expected by OpenRLHF.
    
    This bridges your medical prompt generation to OpenRLHF's message format.
    """
    # Use your existing prompt generation logic
    if player_role == "attacker":
        # Create a mock dataset item for prompt generation
        mock_item = {
            "game_category": game_type,
            "error_note": medical_note if "harmful" in game_type else "",
            "clean_note": medical_note if "benign" in game_type else "",
            "error_type": custom_configs.get("error_types", ["dosage"])[0] if custom_configs else "dosage"
        }
        
        # Use your build_attacker_prompts logic
        from datasets import Dataset
        ds = Dataset.from_list([mock_item])
        few_shot = Dataset.from_list([])  # Empty for now
        
        prompt_ds = build_attacker_prompts(ds, few_shot, tokenizer, num_shots=0)
        return prompt_ds[0]["prompt"]
        
    else:  # assessor
        # Use your make_assessor_prompts logic
        records = [{
            "original": medical_note,
            "attacked": history[-1]["content"] if history else medical_note,
            "game_category": game_type
        }]
        
        assessor_ds = make_assessor_prompts(records, tokenizer)
        return assessor_ds[0]["prompt"]


def medical_cot_format_check_and_extract(response: str):
    """
    Adapter function that uses your existing CoT format checking.
    """
    return parse_response(response)


def get_medical_cot_formatting_reward(cot_format_violation: bool) -> float:
    """
    Adapter function for CoT formatting rewards.
    """
    return 1.0 if not cot_format_violation else -1.0


# Export the functions that OpenRLHF expects
__all__ = [
    'get_medical_game_reward_general_sum',
    'convert_medical_game_history_to_messages', 
    'medical_cot_format_check_and_extract',
    'get_medical_cot_formatting_reward'
]