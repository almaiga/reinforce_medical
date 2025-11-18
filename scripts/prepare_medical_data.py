#!/usr/bin/env python3
"""
Prepare Medical Data for Training

Simple script that reuses existing data loading logic from script/selfplay/data.py
and optionally saves to JSONL format for OpenRLHF compatibility.

This is just a thin wrapper - all the real logic is in script/selfplay/data.py
"""

import argparse
import sys
import os
from pathlib import Path

# Add script/selfplay to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script', 'selfplay'))

from data import load_and_prepare_data


def save_to_jsonl(dataset, output_path: str):
    """Save HuggingFace Dataset to JSONL format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # HuggingFace datasets have a built-in to_json method!
    dataset.to_json(output_file, orient='records', lines=True)
    
    print(f"✅ Saved {len(dataset)} records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare medical data using existing data loading logic"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=400,
        help="Total number of samples (split 25/25/25/25 across 4 game types)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/medical_test_4way",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to disk, just validate data loading"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Data Preparation")
    print("="*60)
    print(f"Using existing data loader from script/selfplay/data.py")
    print(f"Num samples: {args.num_samples}")
    print("="*60 + "\n")
    
    # Load data using existing function
    # This already does everything: loads MEDEC, creates 4-way split, etc.
    # Note: We get few_shot back but don't use it since models are SFT'd first
    ds_seeds, ds_few_shot = load_and_prepare_data(args.num_samples)
    
    print(f"\n📊 Data loaded successfully:")
    print(f"   - Seeds dataset: {len(ds_seeds)} samples")
    print(f"   - Few-shot examples: {len(ds_few_shot)} samples (not used - models are SFT'd)")
    print(f"\n   Columns in seeds: {ds_seeds.column_names}")
    
    # Show distribution
    if "game_category" in ds_seeds.column_names:
        from collections import Counter
        categories = Counter(ds_seeds["game_category"])
        print(f"\n   Game category distribution:")
        for cat, count in categories.items():
            print(f"      - {cat}: {count}")
    
    # Show example
    print(f"\n📋 Example record:")
    print(f"   Game category: {ds_seeds[0]['game_category']}")
    print(f"   Error type: {ds_seeds[0]['error_type']}")
    print(f"   Error note (first 100 chars): {ds_seeds[0]['error_note'][:100]}...")
    print(f"   Clean note (first 100 chars): {ds_seeds[0]['clean_note'][:100]}...")
    
    if args.no_save:
        print("\n✅ Data validation complete (--no-save flag set)")
        return 0
    
    # Save to JSONL (no train/val split - you have separate test file)
    output_path = f"{args.output_dir}/selfplay_data.jsonl"
    
    save_to_jsonl(ds_seeds, output_path)
    
    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)
    print(f"\nOutput file:")
    print(f"   - {output_path}")
    print(f"\nNote: No train/val split (you have separate test file)")
    print(f"Note: Few-shot examples not saved (models are SFT'd first)")
    
    return 0


if __name__ == "__main__":
    exit(main())
