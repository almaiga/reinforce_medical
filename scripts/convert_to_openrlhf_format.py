#!/usr/bin/env python3
"""
Convert Medical RL Training Data to OpenRLHF REINFORCE++ Format

Converts the medical training data from our format to the format expected
by OpenRLHF's RedTeamGamePromptDataset.

Input format (our current):
{
    "game_category": "vanilla_harmful" | "adversarial_harmful" |
                     "vanilla_benign" | "adversarial_benign",
    "error_note": "medical note with error",
    "clean_note": "corrected medical note",
    "error_type": "diagnosis" | "management" | etc.
}

Output format (OpenRLHF expected):
{
    "vanilla": "prompt text",
    "adversarial": "prompt text",
    "completion": "expected completion (for vanilla_benign only)",
    "data_type": "vanilla_harmful" | "adversarial_harmful" |
                 "vanilla_benign" | "adversarial_benign"
}

Mapping logic:
- vanilla_harmful: vanilla=error_note, adversarial="", completion=""
- adversarial_harmful: vanilla="", adversarial=error_note, completion=""
- vanilla_benign: vanilla=clean_note, adversarial="", completion="Safe"
- adversarial_benign: vanilla="", adversarial=clean_note, completion=""
"""

import argparse
import json
from pathlib import Path


def convert_record(record: dict) -> dict:
    """Convert a single record to OpenRLHF format."""
    game_category = record["game_category"]
    error_note = record["error_note"]
    clean_note = record["clean_note"]
    error_type = record.get("error_type", "unknown")

    # Initialize output record
    output = {
        "vanilla": "",
        "adversarial": "",
        "completion": "",
        "data_type": game_category
    }

    # Map based on game category
    if game_category == "vanilla_harmful":
        # Vanilla harmful: use error note directly
        output["vanilla"] = error_note
    elif game_category == "adversarial_harmful":
        # Adversarial harmful: error note as seed for attacker
        output["adversarial"] = error_note
    elif game_category == "vanilla_benign":
        # Vanilla benign: use clean note directly with "Safe" completion
        output["vanilla"] = clean_note
        output["completion"] = "Safe"
    elif game_category == "adversarial_benign":
        # Adversarial benign: clean note as seed for attacker
        output["adversarial"] = clean_note
    else:
        raise ValueError(f"Unknown game_category: {game_category}")

    return output


def convert_file(input_path: str, output_path: str):
    """Convert entire JSONL file."""
    print(f"Converting {input_path} to {output_path}")

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert records
    converted_records = []
    category_counts = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            converted = convert_record(record)
            converted_records.append(converted)

            # Track categories
            cat = converted["data_type"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in converted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Log statistics
    print(f"\n✅ Converted {len(converted_records)} records")
    print(f"📊 Distribution:")
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        pct = 100 * count / len(converted_records)
        print(f"   - {cat}: {count} ({pct:.1f}%)")

    print(f"\n✅ Saved to: {output_path}")

    # Show example
    print(f"\n📋 Example record:")
    print(json.dumps(converted_records[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Convert medical data to OpenRLHF REINFORCE++ format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/medical_rl_training/train.jsonl",
        help="Input JSONL file (our format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/medical_openrlhf/train.jsonl",
        help="Output JSONL file (OpenRLHF format)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Convert to OpenRLHF REINFORCE++ Format")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")

    convert_file(args.input, args.output)

    print("\n" + "="*60)
    print("✅ Conversion Complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
