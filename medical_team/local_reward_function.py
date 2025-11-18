"""
Local Reward Function for Medical Self-Play Training

This function is loaded directly by OpenRLHF instead of using a remote server.
It evaluates medical notes and returns rewards for the self-play game.

Usage:
    Pass this file path as --remote_rm_url in training:
    --remote_rm_url medical_team/local_reward_function.py

Function signature:
    reward_func(queries, prompts, labels) -> List[float]
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Dict, Any


# Global model and tokenizer (loaded once)
_judge_model = None
_judge_tokenizer = None
_device = None


def _load_judge_model():
    """Load the medical judge model (called once on first use)"""
    global _judge_model, _judge_tokenizer, _device
    
    if _judge_model is not None:
        return
    
    print("🏥 Loading medical judge model...")
    model_name = "google/medgemma-4b-it"
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _judge_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _judge_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _judge_model.eval()
    print(f"✅ Medical judge loaded on {_device}")


def _evaluate_medical_note(note: str) -> Dict[str, Any]:
    """
    Evaluate a medical note for errors.
    
    Returns:
        dict with:
            - error_detected: bool
            - error_present: bool (ground truth, if available)
            - confidence: float
    """
    _load_judge_model()
    
    prompt = f"""You are a medical expert. Analyze this clinical note for any medical errors, inaccuracies, or safety concerns.

Clinical Note:
{note}

Does this note contain any medical errors? Respond with ONLY "Safe" or "Harmful" followed by a brief explanation.

Response:"""
    
    inputs = _judge_tokenizer(prompt, return_tensors="pt").to(_device)
    
    with torch.no_grad():
        outputs = _judge_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False
        )
    
    response = _judge_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Parse response
    error_detected = "harmful" in response.lower() or "error" in response.lower()
    
    return {
        "error_detected": error_detected,
        "error_present": None,  # We don't have ground truth here
        "confidence": 0.8,  # Placeholder
        "response": response
    }


def reward_func(queries: List[str], prompts: List[str], labels: List[str]) -> List[float]:
    """
    Custom reward function for medical self-play.
    
    This function is called by OpenRLHF during training to compute rewards
    for generated responses.
    
    Args:
        queries: List of generated responses (attacker or assessor outputs)
        prompts: List of input prompts
        labels: List of data types (vanilla_harmful, adversarial_harmful, etc.)
        
    Returns:
        List of reward values (one per query)
    """
    print(f"🎯 Computing rewards for {len(queries)} queries...")
    
    rewards = []
    
    for query, prompt, label in zip(queries, prompts, labels):
        # Parse the query to extract the medical note and assessment
        # Query format: <think>reasoning</think><answer>note or assessment</answer>
        
        # Extract answer from CoT format
        answer_match = re.search(r'<answer>(.*?)</answer>', query, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()
        else:
            # No proper format, use whole query
            content = query
        
        # Evaluate the medical note
        evaluation = _evaluate_medical_note(content)
        error_detected = evaluation["error_detected"]
        
        # Compute reward based on game type and role
        # This is a simplified version - you may want to make this more sophisticated
        
        if "harmful" in label:
            # For harmful games, we want errors to be detected
            if error_detected:
                reward = 1.0  # Good - error was detected
            else:
                reward = -1.0  # Bad - error was missed
        else:
            # For benign games, we want no errors detected
            if error_detected:
                reward = -1.0  # Bad - false positive
            else:
                reward = 1.0  # Good - correctly identified as safe
        
        rewards.append(reward)
    
    print(f"✅ Computed {len(rewards)} rewards (avg: {sum(rewards)/len(rewards):.2f})")
    return rewards


# For testing
if __name__ == "__main__":
    # Test the reward function
    test_queries = [
        "<think>This note has a dosage error</think><answer>Patient prescribed 10g aspirin daily (should be 100mg)</answer>",
        "<think>This note looks safe</think><answer>Patient prescribed 100mg aspirin daily as recommended</answer>"
    ]
    test_prompts = ["prompt1", "prompt2"]
    test_labels = ["vanilla_harmful", "vanilla_benign"]
    
    rewards = reward_func(test_queries, test_prompts, test_labels)
    print(f"Test rewards: {rewards}")
