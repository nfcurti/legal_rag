#!/usr/bin/env python3
"""
Precompute Swiss legal corpus embeddings and save them as a .npy file.

Run this *offline* (locally or on a throwaway EC2 with PyTorch installed), not on
the small production instance. Example:

  cd python_backend
  python precompute_embeddings.py --split val --max-cases 5000

By default writes:
  data/swiss_legal/embeddings_val.npy

Set LEGAL_EMBEDDINGS_PATH to override the output path.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_prediction import _load_swiss_legal_corpus


def main() -> None:
  parser = argparse.ArgumentParser(description="Precompute Swiss legal corpus embeddings to .npy")
  parser.add_argument("--split", default="val", help="Dataset split to use (val/train)")
  parser.add_argument("--max-cases", type=int, default=5_000, help="Maximum number of cases to embed")
  parser.add_argument(
    "--model",
    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    help="SentenceTransformer model name",
  )
  parser.add_argument(
    "--out",
    default="",
    help="Output .npy path (default: DATA_DIR/embeddings_<split>.npy or LEGAL_EMBEDDINGS_PATH)",
  )
  args = parser.parse_args()

  # Load corpus using the same helper as runtime code, but without S3 streaming.
  corpus = _load_swiss_legal_corpus(split=args.split, max_cases=args.max_cases)
  if not corpus:
    raise SystemExit(f"No cases loaded for split={args.split}")

  texts = [c["facts"] for c in corpus]
  print(f"Loaded {len(texts)} cases; embedding with {args.model}...")

  model = SentenceTransformer(args.model)
  emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
  emb = np.asarray(emb, dtype=np.float32)

  # Decide output path
  embeddings_path_env = os.environ.get("LEGAL_EMBEDDINGS_PATH", "").strip()
  if args.out:
    out_path = Path(args.out)
  elif embeddings_path_env:
    out_path = Path(embeddings_path_env)
  else:
    root = Path(__file__).resolve().parent
    out_path = root / "data" / "swiss_legal" / f"embeddings_{args.split}.npy"

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(out_path, emb)
  print(f"Saved embeddings with shape {emb.shape} to {out_path}")


if __name__ == "__main__":
  main()

