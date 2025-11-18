#!/usr/bin/env python3
"""
Create RL Training Data from MEDEC

Creates 100% training data (no validation split) for RL training.
Uses 4-way game structure for medical self-play.
"""

import argparse
import pandas as pd
from pathlib import Path
import json


def load_medec_data(csv_path: str) -> pd.DataFrame:
    """Load MEDEC dataset from CSV."""
    print(f"Loading MEDEC data from: {csv_path}")
    df = pd.read_csv(csv_path).fillna("")

    # Filter to only rows with errors (Error Flag = 1)
    df_errors = df[df["Error Flag"] == 1].copy()

    # Filter out empty text
    df_errors = df_errors[
        (df_errors["Text"].str.strip() != "") &
        (df_errors["Corrected Text"].str.strip() != "")
    ]

    print(f"✅ Loaded {len(df_errors)} error cases from MEDEC")
    print(f"📊 Error types present: "
          f"{df_errors['Error Type'].nunique()} unique types")
    return df_errors


def create_4way_split(
    df: pd.DataFrame,
    num_samples: int = None
) -> pd.DataFrame:
    """
    Create 4-way game structure split.

    Each MEDEC error case has BOTH error_note and clean_note.
    We create 2 data points from EACH case:

    1. One for harmful game (using error_note) - vanilla or adversarial
    2. One for benign game (using clean_note) - vanilla or adversarial

    Result: 2x data points from N error cases = 2N total data points
    With 25% in each category.
    """
    if num_samples is None:
        num_samples = len(df)

    # Shuffle for random distribution
    df_shuffled = df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Ensure we have enough data
    if len(df_shuffled) < num_samples:
        print(f"⚠️  Only {len(df_shuffled)} samples available, using all")
        num_samples = len(df_shuffled)

    # Take the requested number of samples
    df_subset = df_shuffled.head(num_samples).copy()

    datasets = []

    # For each error case, create 2 data points
    for idx, row in df_subset.iterrows():
        # Determine which harmful and benign category for this case
        # Alternate to get 50/50 split
        if idx % 2 == 0:
            harmful_cat = "vanilla_harmful"
            benign_cat = "vanilla_benign"
        else:
            harmful_cat = "adversarial_harmful"
            benign_cat = "adversarial_benign"

        # Create harmful data point (uses error_note)
        row_harmful = row.copy()
        row_harmful["game_category"] = harmful_cat
        datasets.append(row_harmful)

        # Create benign data point (uses clean_note)
        row_benign = row.copy()
        row_benign["game_category"] = benign_cat
        datasets.append(row_benign)

    # Convert list of series to dataframe
    df_combined = pd.DataFrame(datasets)

    # Log distribution
    print(f"\n✅ Created 4-way game structure from "
          f"{num_samples} error cases:")
    print(f"   Total data points: {len(df_combined)} (2x original)")
    categories = [
        "vanilla_harmful", "adversarial_harmful",
        "vanilla_benign", "adversarial_benign"
    ]
    for category in categories:
        count = (df_combined["game_category"] == category).sum()
        pct = (100 * count / len(df_combined)
               if len(df_combined) > 0 else 0)
        print(f"   - {category}: {count} ({pct:.1f}%)")

    return df_combined


def convert_to_jsonl_format(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSONL format for RL training."""
    records = []

    for _, row in df.iterrows():
        game_category = row["game_category"]
        error_note = row["Text"].strip()
        clean_note = row["Corrected Text"].strip()
        error_type = row.get("Error Type", "unknown")

        record = {
            "game_category": game_category,
            "error_note": error_note,
            "clean_note": clean_note,
            "error_type": error_type
        }

        records.append(record)

    # Log error type distribution
    error_types = {}
    for record in records:
        et = record["error_type"]
        error_types[et] = error_types.get(et, 0) + 1

    print("\n📊 Error type distribution:")
    for et, count in sorted(error_types.items()):
        pct = 100 * count / len(records) if len(records) > 0 else 0
        print(f"   - {et}: {count} ({pct:.1f}%)")

    return records


def save_jsonl(records: list, output_path: str):
    """Save records to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n✅ Saved {len(records)} records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create RL training data from MEDEC "
            "(100% training, no validation)"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=(
            "data_copy/MEDEC/MEDEC-MS/"
            "MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
        ),
        help="Path to MEDEC CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/medical_rl_training",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all available)"
    )

    args = parser.parse_args()

    print("="*60)
    print("MEDEC to RL Training Data")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    samples_msg = (
        'All available'
        if args.num_samples is None
        else args.num_samples
    )
    print(f"Samples: {samples_msg}")
    print("="*60 + "\n")

    # Load data
    df = load_medec_data(args.input)

    # Create 4-way split
    df_split = create_4way_split(df, args.num_samples)

    # Convert to JSONL format
    records = convert_to_jsonl_format(df_split)

    # Save (100% training, no validation)
    train_path = f"{args.output_dir}/train.jsonl"
    save_jsonl(records, train_path)

    print("\n" + "="*60)
    print("✅ RL Training Data Created!")
    print("="*60)
    print(f"\nOutput file: {train_path}")
    print(f"Total samples: {len(records)}")
    print("\n📋 Example record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    exit(main())
