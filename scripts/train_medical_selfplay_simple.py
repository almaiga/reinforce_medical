#!/usr/bin/env python3
"""
Simple Medical Self-Play Training Script

Inspired by Self-RedTeam's approach but simplified for single-GPU training.
This script demonstrates the core self-play loop without full OpenRLHF Ray complexity.

For full distributed training, adapt OpenRLHF's train_ppo_ray.py
"""

import argparse
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script', 'selfplay'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data import load_and_prepare_data
from medical_team import (
    MedicalDialogueGameManager,
    create_medical_judge_remote_function,
    test_judge_connection
)


class SimpleStrategy:
    """Simple strategy for single-GPU training (replaces Ray strategy)"""
    def __init__(self, rank=0):
        self.rank = rank
    
    def get_rank(self):
        return self.rank
    
    def print(self, msg):
        if self.rank == 0:
            print(msg)


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer


def create_llm_generator(model, tokenizer, device):
    """Create LLM generator function"""
    def generator(prompts, labels):
        outputs = []
        
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(
                output_ids[0, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Create mock output object (for compatibility)
            class MockOutput:
                def __init__(self, text):
                    self.text = text
                    self.outputs = [self]
                    self.token_ids = output_ids[0].tolist()
            
            outputs.append(MockOutput(response))
        
        return outputs
    
    return generator


def run_self_play_episode(
    game_manager,
    attacker_generator,
    assessor_generator,
    episode_num
):
    """Run one self-play episode"""
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}")
    print(f"{'='*60}")
    
    # Play games
    print("Playing games...")
    results = game_manager.play_games(
        attacker_llm_generator=attacker_generator,
        assessor_llm_generator=assessor_generator
    )
    
    print(f"Completed {len(results)} games")
    
    # Evaluate with judge
    print("Evaluating with judge...")
    batch_labels = game_manager.evaluate_game_outcomes()
    
    # Compute rewards
    print("Computing rewards...")
    attacker_outputs, attacker_states, assessor_outputs, assessor_states = \
        game_manager.filter_and_compute_rewards(batch_labels)
    
    # Log statistics
    print(f"\nResults:")
    print(f"  Attacker samples: {len(attacker_outputs)}")
    print(f"  Assessor samples: {len(assessor_outputs)}")
    
    if attacker_states:
        avg_attacker_reward = sum(s['reward'] for s in attacker_states) / len(attacker_states)
        print(f"  Avg attacker reward: {avg_attacker_reward:.3f}")
    
    if assessor_states:
        avg_assessor_reward = sum(s['reward'] for s in assessor_states) / len(assessor_states)
        print(f"  Avg assessor reward: {avg_assessor_reward:.3f}")
    
    return {
        'attacker_outputs': attacker_outputs,
        'attacker_states': attacker_states,
        'assessor_outputs': assessor_outputs,
        'assessor_states': assessor_states,
        'batch_labels': batch_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simple medical self-play training (single GPU)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to train"
    )
    parser.add_argument(
        "--judge-url",
        type=str,
        default="http://localhost:8000",
        help="Judge server URL (start with scripts/serve_medical_judge.py)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="Number of samples per episode"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of self-play episodes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/medical_selfplay",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Self-Play Training (Simple)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Judge URL: {args.judge_url}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num episodes: {args.num_episodes}")
    print("="*60)
    
    # Check judge connection
    print("\n🔍 Checking judge server...")
    if not test_judge_connection(args.judge_url):
        print("❌ Judge server not available!")
        print(f"   Start it with: python scripts/serve_medical_judge.py")
        return 1
    
    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Create generators
    print("\n🎮 Creating generators...")
    attacker_generator = create_llm_generator(model, tokenizer, args.device)
    assessor_generator = create_llm_generator(model, tokenizer, args.device)
    
    # Create judge function
    print("\n⚖️  Creating judge function...")
    judge_fn = create_medical_judge_remote_function(args.judge_url)
    
    # Create strategy
    strategy = SimpleStrategy(rank=0)
    
    # Run episodes
    print("\n🚀 Starting self-play episodes...")
    
    for episode in range(args.num_episodes):
        # Load fresh data for each episode
        print(f"\n📊 Loading data for episode {episode+1}...")
        ds_seeds, _ = load_and_prepare_data(args.num_samples)
        
        # Initialize game manager
        game_manager = MedicalDialogueGameManager(
            tokenizer=tokenizer,
            medical_judge_fn=judge_fn,
            strategy=strategy,
            custom_configs={
                "max_turns": 2,
                "reward_type": "medical_general_sum",
                "error_types": ["dosage", "diagnosis", "contraindication"],
                "direct_chat_no_cot": False
            }
        )
        
        # Initialize games
        medical_notes = ds_seeds["error_note"]
        completions = ["Expected assessment"] * len(ds_seeds)
        data_types = ds_seeds["game_category"]
        
        game_manager.initialize_games(medical_notes, completions, data_types)
        
        # Run episode
        results = run_self_play_episode(
            game_manager,
            attacker_generator,
            assessor_generator,
            episode + 1
        )
        
        # TODO: Here you would normally:
        # 1. Collect experiences for REINFORCE++
        # 2. Update model with policy gradient
        # 3. Save checkpoints
        # 
        # For now, this just demonstrates the self-play loop
    
    print("\n" + "="*60)
    print("✅ Self-play episodes complete!")
    print("="*60)
    print("\nNote: This is a demonstration script.")
    print("For actual training, integrate with:")
    print("  - TRL's GRPOTrainer (your current approach)")
    print("  - OpenRLHF's train_ppo_ray.py (for distributed)")
    
    return 0


if __name__ == "__main__":
    exit(main())
