#!/usr/bin/env python3
"""
Precompute embeddings for a JSONL file of labeled Swiss legal cases.

This script is intended for one-off generation of a precomputed embeddings
matrix (e.g. v1.npy) that matches exactly the corpus used at runtime.

It:
- Reads a JSONL file (one record per line, like labels (1).jsonl)
- Uses legal_prediction._record_to_case to filter/normalize records
  (including using `url_text` when present, falling back to `facts`)
- Embeds the resulting texts with SentenceTransformer
- Saves a NumPy array (float32) to the given output .npy path

Example:

  cd python_backend
  python precompute_embeddings_from_labels.py \
    --jsonl "/Users/nicolascurti/Downloads/labels (1).jsonl" \
    --out v1.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_prediction import _record_to_case


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute embeddings from a JSONL labels file to .npy"
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to labels JSONL file (one record per line)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--out",
        default="v1.npy",
        help="Output .npy path (default: v1.npy in python_backend)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")

    cases = []
    texts: list[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = _record_to_case(rec)
            if not case:
                continue
            cases.append(case)
            texts.append(case["facts"])

    if not texts:
        raise SystemExit(f"No usable cases found in {jsonl_path}")

    print(f"Loaded {len(texts)} cases from {jsonl_path}; embedding with {args.model}...")

    model = SentenceTransformer(args.model)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        # Save relative to this script's directory (python_backend)
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)

    print(f"Saved embeddings with shape {emb.shape} to {out_path}")


if __name__ == "__main__":
    main()

