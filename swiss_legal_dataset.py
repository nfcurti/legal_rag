#!/usr/bin/env python3
"""
Load Swiss legal cases from Hugging Face and prepare for prediction + reasoning.

Dataset: rcds/swiss_criticality_prediction (139k Swiss Federal Supreme Court cases)
- Prediction: bge_label (critical / non-critical), citation_label (5 classes)
- Reasoning: court's "considerations" (legal reasoning) + "rulings" (outcome)

Output: JSONL with fields for model input (facts), prediction target (labels),
and reasoning (considerations, rulings) for training or inference.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "rcds/swiss_criticality_prediction"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "swiss_legal"


def _sanitize(s):
    if s is None:
        return ""
    return (s or "").strip()


def prepare_for_prediction_and_reasoning(example: dict) -> dict:
    """One record: input (facts), prediction target (labels), reasoning (considerations + rulings)."""
    return {
        "decision_id": example.get("decision_id", ""),
        "language": example.get("language", ""),
        "year": example.get("year"),
        "chamber": example.get("chamber", ""),
        "law_area": example.get("law_area", ""),
        "law_sub_area": _sanitize(example.get("law_sub_area")),
        # Prediction targets
        "bge_label": example.get("bge_label", ""),
        "citation_label": example.get("citation_label", ""),
        # Input: facts (model sees this to predict + reason)
        "facts": _sanitize(example.get("facts")),
        # Reasoning: court's legal reasoning and outcome (ground truth for reasoning)
        "considerations": _sanitize(example.get("considerations")),
        "rulings": _sanitize(example.get("rulings")),
    }


def main():
    parser = argparse.ArgumentParser(description="Load Swiss legal dataset and export for prediction + reasoning")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--law-area", type=str, default=None, help="Filter by law_area (e.g. civil_law)")
    parser.add_argument("--max-train", type=int, default=None, help="Max train samples (for quick experiments)")
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Export format")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from Hugging Face...")
    # Requires datasets<4 (loading scripts deprecated in 4.x)
    ds = load_dataset(DATASET_ID, trust_remote_code=True)

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        data = ds[split]
        max_n = {"train": args.max_train, "validation": args.max_val, "test": args.max_test}.get(split)

        rows = []
        for i, ex in enumerate(data):
            if max_n is not None and i >= max_n:
                break
            if args.law_area and ex.get("law_area") != args.law_area:
                continue
            rows.append(prepare_for_prediction_and_reasoning(ex))

        if not rows:
            print(f"  {split}: 0 rows (skipped)")
            continue

        out_name = "train" if split == "train" else ("val" if split == "validation" else "test")
        if args.format == "jsonl":
            out_path = args.output_dir / f"{out_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            out_path = args.output_dir / f"{out_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"  {split}: {len(rows)} rows -> {out_path}")

    # Write a small schema/readme for prediction + reasoning usage
    readme = args.output_dir / "README.txt"
    readme.write_text("""# Swiss Legal Cases – Prediction + Reasoning

Source: Hugging Face rcds/swiss_criticality_prediction (Swiss Federal Supreme Court).

Fields per record:
- decision_id, language, year, chamber, law_area, law_sub_area
- bge_label: binary (critical / non-critical)
- citation_label: 5 classes (critical-1 .. critical-4, non-critical)
- facts: case facts (INPUT for prediction/reasoning)
- considerations: court's legal reasoning (REASONING target)
- rulings: court's ruling (OUTCOME)

Usage for prediction: given `facts` (and optionally part of `considerations`), predict `bge_label` or `citation_label`.
Usage for reasoning: given `facts`, generate `considerations` and/or predict `rulings`; compare to ground truth.
""", encoding="utf-8")
    print(f"  README -> {readme}")


if __name__ == "__main__":
    main()
