#!/usr/bin/env python3
"""
Evaluate the legal prediction pipeline on gold cases (held-out test set).

Loads data/swiss_legal/gold.jsonl. For each case, uses `facts` as the case context
(case_context=facts) and runs predict_with_reasoning. Compares model output
to gold `rulings` and `considerations`. Gold is never used for RAG or training.

Usage:
  # RAG + OpenAI
  python eval_gold.py

  # RAG + fine-tuned model
  python eval_gold.py --model-path ./out_legal_7b

  # Save results for later analysis
  python eval_gold.py --output results.jsonl --max 5
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "swiss_legal" / "gold.jsonl"

# Load .env so OPENAI_API_KEY is set when running from project root
_env_file = ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"").strip()
            if k and v:
                os.environ.setdefault(k, v)


def load_gold(path: Path, max_cases: Optional[int] = None) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_cases is not None and len(cases) >= max_cases:
                break
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on gold cases (held-out test set)")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, help="Path to gold.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Write results to JSONL (one record per case)")
    parser.add_argument("--max", type=int, default=None, help="Max number of gold cases (for quick runs)")
    parser.add_argument("--top-k", type=int, default=5, help="RAG top-k precedents")
    parser.add_argument("--no-dataset", action="store_true", help="Do not use RAG retrieval")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model if not using --model-path")
    parser.add_argument("--model-path", type=str, default=None, help="Path to fine-tuned adapter")
    parser.add_argument("--tfidf", action="store_true", help="Use TF-IDF retrieval instead of embedding RAG")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model for RAG (default sentence-transformers; or BAAI/bge-m3 for FlagEmbedding)")
    parser.add_argument("--max-corpus", type=int, default=5000, help="Max cases to load for RAG corpus (default 5000; lower if OOM on Mac)")
    parser.add_argument("--quiet", action="store_true", help="Only print summary and errors")
    args = parser.parse_args()

    if not args.model_path and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Either:", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...   (then run again)", file=sys.stderr)
        print("  or use a local model: python eval_gold.py --model-path ./out_legal_7b", file=sys.stderr)
        sys.exit(1)

    if not args.gold.exists():
        print(f"Gold file not found: {args.gold}", file=sys.stderr)
        print("Create data/swiss_legal/gold.jsonl with one JSON object per line (facts, considerations, rulings).", file=sys.stderr)
        sys.exit(1)

    from legal_prediction import predict_with_reasoning

    cases = load_gold(args.gold, args.max)
    if not cases:
        print("No cases in gold file.", file=sys.stderr)
        sys.exit(1)
    print(f"Evaluating on {len(cases)} gold case(s). Gold is not used for RAG or training.\n")

    results = []
    for i, rec in enumerate(cases):
        decision_id = rec.get("decision_id", f"case-{i+1}")
        facts = (rec.get("facts") or "").strip()
        gold_rulings = (rec.get("rulings") or "").strip()
        gold_considerations = (rec.get("considerations") or "").strip()
        if not facts or (not gold_rulings and not gold_considerations):
            print(f"Skip {decision_id}: missing facts or gold output.", file=sys.stderr)
            continue

        # Use facts as case context (gold has no separate claim/defense)
        try:
            out = predict_with_reasoning(
                facts,
                top_k=args.top_k,
                use_dataset=not args.no_dataset,
                use_embeddings=not args.tfidf,
                embedding_model=args.embedding_model,
                max_cases=args.max_corpus,
                model=args.model,
                model_path=args.model_path,
            )
        except Exception as e:
            print(f"{decision_id}: error — {e}", file=sys.stderr)
            results.append({
                "decision_id": decision_id,
                "error": str(e),
                "gold_rulings": gold_rulings,
                "applicable_law": gold_considerations,
                "model_prediction": None,
                "model_reasoning": None,
            })
            continue

        pred = (out.get("prediction") or "").strip()
        reasoning = (out.get("reasoning") or "").strip()
        row = {
            "decision_id": decision_id,
            "gold_rulings": gold_rulings,
            "applicable_law": gold_considerations,
            "model_prediction": pred,
            "model_reasoning": reasoning,
        }
        # Include retrieved precedents with case_id (use for later query/lookup) and decision_id
        precs = out.get("precedent_cases")
        if precs:
            row["retrieved_precedents"] = [
                {
                    "case_id": c.get("decision_id", ""),
                    "decision_id": c.get("decision_id", ""),
                    "chamber": c.get("chamber"),
                    "law_area": c.get("law_area"),
                    "year": c.get("year"),
                    "facts_excerpt": (c.get("facts") or "")[:400],
                    "rulings_excerpt": (c.get("rulings") or "")[:300],
                }
                for c in precs
            ]
        results.append(row)

        if not args.quiet:
            print(f"——— {decision_id} ———")
            print("Gold ruling:", gold_rulings[:200] + ("..." if len(gold_rulings) > 200 else ""))
            print("Model pred: ", pred[:200] + ("..." if len(pred) > 200 else ""))
            print("Gold applicable law (excerpt):", gold_considerations[:150] + ("..." if len(gold_considerations) > 150 else ""))
            print("Model reasoning (excerpt):", reasoning[:150] + ("..." if len(reasoning) > 150 else ""))
            print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} result(s) to {args.output}")

    print(f"Done. {len(results)} case(s) evaluated.")


if __name__ == "__main__":
    main()
