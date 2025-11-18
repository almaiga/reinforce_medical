"""Medical utilities for self-play training - adapted from red_team/utils.py"""

import re
from typing import Tuple, Dict, Any
from medical_team.prompts import (
    MEDICAL_ATTACKER_BASE_PROMPT_HARMFUL,
    MEDICAL_ATTACKER_BASE_PROMPT_BENIGN,
    MEDICAL_ATTACKER_BASE_PROMPT_VANILLA,
    MEDICAL_ASSESSOR_BASE_PROMPT,
    MEDICAL_ASSISTANT_THINKING_PREFIX
)
from medical_team import MedicalGameOutcome

# Medical reward coefficient configuration
MEDICAL_REWARD_COEFF_CONFIG = {
    "cot_format_violation": (-1.0, 1.0),
    "medical_general_sum": {
        "attacker": {
            # NEG, POS - Attacker rewarded when errors go undetected
            "error_detected": (-1.0, 1.0),  # NEG when error detected, POS when undetected
            "error_realism": (-0.5, 0.5),   # NEG for unrealistic errors, POS for realistic
        },
        "assessor": {
            # NEG, POS - Assessor rewarded for correct error detection
            "error_detection": (-1.0, 1.0),  # NEG for missed/false positive, POS for correct
            "explanation_quality": (-0.5, 0.5),  # NEG for poor explanation, POS for good
        }
    }
}

def convert_medical_game_history_to_messages(history, player_role, medical_note, game_type, custom_configs=None, tokenizer=None):
    """
    Convert medical game history to messages for model input
    
    Args:
        history: Game history so far
        player_role: "attacker" or "assessor" 
        medical_note: Original medical note
        error_type: Type of error to introduce/detect
        custom_configs: Custom configuration options
        tokenizer: Tokenizer for chat template formatting
    """
    messages = convert_medical_game_history_to_query(history, player_role, medical_note, game_type, custom_configs)
    
    chat_message = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Add CoT prefix if not disabled
    if custom_configs and custom_configs.get("direct_chat_no_cot", False):
        pass
    else:
        chat_message += MEDICAL_ASSISTANT_THINKING_PREFIX
    
    return chat_message

def convert_medical_game_history_to_query(history, player_role, medical_note, game_type, custom_configs=None):
    """
    Convert medical game history to query messages
    
    Args:
        history: Game history so far
        player_role: "attacker" or "assessor"
        medical_note: Original medical note
        error_type: Type of error to introduce/detect
        custom_configs: Custom configuration options
    """
    messages = []
    
    # Set system prompt and instruction based on role
    if player_role == "attacker":
        # Get system prompt based on game type
        if game_type == "harmful":
            system_prompt = MEDICAL_ATTACKER_BASE_PROMPT_HARMFUL
            user_query = f"Create an error variation of this seed note:\n\n{medical_note}\n\nFormat:\n<think>your substitution strategy</think>\n<output>note with ONE error</output>"
        elif game_type == "benign":
            system_prompt = MEDICAL_ATTACKER_BASE_PROMPT_BENIGN
            user_query = f"Verify this medical note is safe and output it unchanged:\n\n{medical_note}"
        else:  # vanilla
            system_prompt = MEDICAL_ATTACKER_BASE_PROMPT_VANILLA
            user_query = f"Verify this medical note is safe and output it unchanged:\n\n{medical_note}"
        
    else:  # assessor
        system_prompt = MEDICAL_ASSESSOR_BASE_PROMPT
        user_query = f"Classify this medical note for errors:\n\n{history[0]['content']}"
    
    # Add system message
    messages.append({"role": "system", "content": system_prompt})
    
    # Add user query
    messages.append({"role": "user", "content": user_query})
    
    return messages

def get_medical_cot_formatting_reward(cot_format_violation: bool) -> float:
    """Get reward for CoT formatting compliance"""
    cot_NEG, cot_POS = MEDICAL_REWARD_COEFF_CONFIG["cot_format_violation"]
    if cot_format_violation:
        return cot_NEG
    else:
        return cot_POS

def medical_cot_format_check_and_extract(response: str) -> Tuple[Tuple[str, str], bool]:
    """
    Check if medical response follows correct CoT format and extract thinking/answer
    
    Returns:
        ((thinking, answer), format_violation_flag)
    """
    THINK_PATTERN = re.compile(r"\s*(.*?)\s*</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    
    try:
        # Extract thinking and answer content
        thinking = THINK_PATTERN.search(response).group(1).strip()
        answer = ANSWER_PATTERN.search(response).group(1).strip()
        
        # Check if <think> appears (it shouldn't since it's prepended)
        if "<think>" in response:
            return (None, answer), True
            
        # Check for empty thinking or answer
        if not thinking or not answer:
            return (None, answer), True
        
        # Count tag occurrences
        if (response.count('</think>') != 1 or 
            response.count('<answer>') != 1 or 
            response.count('</answer>') != 1):
            return (None, answer), True
        
        # Check correct order of tags
        think_end_pos = response.find('</think>')
        answer_start_pos = response.find('<answer>')
        answer_end_pos = response.find('</answer>')
        
        # Verify tag order: </think> -> <answer> -> </answer>
        if not (0 < think_end_pos < answer_start_pos < answer_end_pos):
            return (None, answer), True
        
        # Check if </think> is immediately followed by <answer>
        between_tags = response[think_end_pos + len('</think>'):answer_start_pos].strip()
        if between_tags:
            return (None, answer), True
        
        # Check if response ends with </answer>
        if not response.strip().endswith('</answer>'):
            return (thinking, answer), True
        
        # Optional: Check for any content after </answer>
        if response[answer_end_pos + len('</answer>'):].strip():
            return (thinking, answer), True
        
        # All checks passed
        return (thinking, answer), False
        
    except (AttributeError, IndexError):
        # Handle cases where regex patterns don't match
        try:
            # Try to extract just the answer if possible
            answer = ANSWER_PATTERN.search(response).group(1).strip()
            return (None, answer), True
        except (AttributeError, IndexError):
            # If we can't even get the answer, return the original response
            return (None, response), True

def get_medical_error_detection_reward(role: str, error_detected: bool, error_present: bool, error_type: str = None, reward_coeff_config: str = "medical_general_sum") -> Tuple[float, set]:
    """
    Calculate reward based on medical error detection accuracy
    
    Args:
        role: "attacker" or "assessor"
        error_detected: Whether the assessor detected an error
        error_present: Whether an error was actually present
        error_type: Type of medical error
        reward_coeff_config: Reward configuration to use
    
    Returns:
        (reward, outcomes_set)
    """
    config = MEDICAL_REWARD_COEFF_CONFIG[reward_coeff_config]
    reward = 0.0
    outcomes = set()
    
    if role == "attacker":
        # Attacker is rewarded when errors go undetected
        r_NEG, r_POS = config["attacker"]["error_detected"]
        
        if error_present:
            if error_detected:
                # Error was detected - bad for attacker
                reward += r_NEG
                outcomes.add(MedicalGameOutcome.ATTACKER_ERROR_DETECTED)
            else:
                # Error went undetected - good for attacker
                reward += r_POS
                outcomes.add(MedicalGameOutcome.ATTACKER_ERROR_UNDETECTED)
        # If no error present, attacker doesn't get reward from this component
        
    elif role == "assessor":
        # Assessor is rewarded for correct detection/rejection
        r_NEG, r_POS = config["assessor"]["error_detection"]
        
        if error_present and error_detected:
            # Correctly detected error
            reward += r_POS
            outcomes.add(MedicalGameOutcome.ASSESSOR_CORRECT_DETECTION)
        elif error_present and not error_detected:
            # Missed error
            reward += r_NEG
            outcomes.add(MedicalGameOutcome.ASSESSOR_MISSED_ERROR)
        elif not error_present and not error_detected:
            # Correctly rejected (no false positive)
            reward += r_POS
            outcomes.add(MedicalGameOutcome.ASSESSOR_CORRECT_REJECTION)
        elif not error_present and error_detected:
            # False positive
            reward += r_NEG
            outcomes.add(MedicalGameOutcome.ASSESSOR_FALSE_POSITIVE)
    
    return reward, outcomes

def get_medical_error_realism_reward(role: str, error_realistic: bool, reward_coeff_config: str = "medical_general_sum") -> Tuple[float, set]:
    """
    Calculate reward based on medical error realism (for attacker)
    
    Args:
        role: "attacker" or "assessor"
        error_realistic: Whether the introduced error is realistic
        reward_coeff_config: Reward configuration to use
    
    Returns:
        (reward, outcomes_set)
    """
    if role != "attacker":
        return 0.0, set()
    
    config = MEDICAL_REWARD_COEFF_CONFIG[reward_coeff_config]
    r_NEG, r_POS = config["attacker"]["error_realism"]
    
    reward = 0.0
    outcomes = set()
    
    if error_realistic:
        reward += r_POS
        outcomes.add(MedicalGameOutcome.ATTACKER_REALISTIC_ERROR)
    else:
        reward += r_NEG
        outcomes.add(MedicalGameOutcome.ATTACKER_UNREALISTIC_ERROR)
    
    return reward, outcomes

def get_medical_game_reward_general_sum(gamedata: Dict[str, Any], labels: Dict[str, Any], reward_coeff_config: str = "medical_general_sum") -> Tuple[float, set]:
    """
    Calculate total reward for medical self-play game
    
    Args:
        gamedata: Game data containing role, error_type, etc.
        labels: Labels from medical judge model containing error detection results
        reward_coeff_config: Reward configuration to use
    
    Returns:
        (total_reward, all_outcomes_set)
    """
    role = gamedata["game_role"]
    error_type = gamedata.get("error_type")
    
    if role not in ("attacker", "assessor"):
        raise ValueError(f"Invalid role: {role}")
    
    total_reward = 0.0
    all_outcomes = set()
    
    # Extract labels from judge model
    error_detected = labels.get("error_detected", False)
    error_present = labels.get("error_present", False)
    error_realistic = labels.get("error_realistic", True)  # Default to realistic
    
    # Calculate error detection reward
    detection_reward, detection_outcomes = get_medical_error_detection_reward(
        role, error_detected, error_present, error_type, reward_coeff_config
    )
    total_reward += detection_reward
    all_outcomes.update(detection_outcomes)
    
    # Calculate error realism reward (only for attacker)
    if role == "attacker":
        realism_reward, realism_outcomes = get_medical_error_realism_reward(
            role, error_realistic, reward_coeff_config
        )
        total_reward += realism_reward
        all_outcomes.update(realism_outcomes)
    
    return total_reward, all_outcomes

def convert_medical_game_history_to_conversation(history, error_type=None):
    """Convert medical game history to readable conversation format"""
    history_str = ""
    if error_type:
        history_str = f"Error type: {error_type}\n"
        history_str += "=" * 30 + "\n"
    
    for message in history:
        role = message['game_role'].title()
        history_str += f"<|{role}|>: {message['content']}"
        history_str += "\n" + "-" * 30 + "\n"
    
    return history_str


# ============================================================================
# Aliases for OpenRLHF compatibility
# OpenRLHF expects these function names from red_team.utils
# ============================================================================

# Alias for CoT format checking
cot_format_check_and_extract = medical_cot_format_check_and_extract

# Alias for CoT formatting reward
get_cot_formatting_reward = get_medical_cot_formatting_reward

# Alias for general sum reward
get_redteaming_game_reward_general_sum = get_medical_game_reward_general_sum

# Zero-sum reward (same as general sum for our case)
def get_redteaming_game_reward_zero_sum(gamedata: Dict[str, Any], labels: Dict[str, Any], reward_coeff_config: str = "medical_general_sum") -> Tuple[float, set]:
    """
    Zero-sum reward variant (alias for general sum in medical domain)
    
    In medical self-play, the rewards are already zero-sum:
    - When attacker succeeds (error undetected), assessor fails
    - When assessor succeeds (error detected), attacker fails
    """
    return get_medical_game_reward_general_sum(gamedata, labels, reward_coeff_config)