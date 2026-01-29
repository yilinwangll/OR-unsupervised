#!/usr/bin/env python3
"""
Data Preparation Script for TRL GRPO Training

This script converts data_selected.json to TRL format with all fields directly accessible.

Input format:
    {
        "en_question": "...",
        "source": "...",
        "label": "MaxFlow|TSP|IntegerConstraint|Other",
        "cloze": "..."
    }

Output format:
    {
        "question": "...",
        "label": "...",
        "cloze": "..."
    }

Usage:
    python prepare_trl_data.py \
        --input /path/to/data_selected.json \
        --output /path/to/grpo_trl_data.json
"""

import json
import argparse
from collections import Counter


def process_data(data):
    """Process data to TRL format with direct field access."""
    processed = []
    for item in data:
        question = item.get("en_question", item.get("question", ""))
        label = item.get("label", "Other")
        cloze = item.get("cloze", "")

        processed.append({
            "question": question.strip(),
            "label": label,
            "cloze": cloze,
        })

    return processed


def main():
    parser = argparse.ArgumentParser(description="Prepare data for TRL GRPO training")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    # Show label distribution
    labels = [item.get("label", "unknown") for item in data]
    print("Label distribution:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

    # Process
    processed = process_data(data)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(processed)} items to {args.output}")


if __name__ == "__main__":
    main()
