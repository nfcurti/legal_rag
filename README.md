# Legal RAG

Retrieval-augmented generation for Swiss Federal Supreme Court case prediction and legal reasoning.

Given a case summary (claim/defense context), the pipeline retrieves similar precedents via multilingual embeddings, then produces an outcome prediction grounded in real court decisions — with cited reasoning, doctrine excerpts, and precedent relevance.

Built for production use: FastAPI async jobs, optional S3-backed corpora, OpenAI or local LLMs, and gold-set evaluation.

## What it does

- **Embedding RAG** over Swiss court cases (`sentence-transformers` / multilingual MiniLM, with TF-IDF fallback)
- **Precedent-grounded generation** — prediction + legal basis + reasoning, not free-form hallucination
- **Async prediction API** (`POST /api/legal-predict` → poll job status or webhook callback)
- **Dataset prep** from Hugging Face [`rcds/swiss_criticality_prediction`](https://huggingface.co/datasets/rcds/swiss_criticality_prediction) (~139k cases)
- **Eval harness** against held-out gold cases (`eval_gold.py`)
- **S3 streaming / sync** for corpora and precomputed embeddings in deploy environments

## Stack

| Layer | Tech |
| --- | --- |
| Retrieval | sentence-transformers, scikit-learn TF-IDF, optional FlagEmbedding |
| Generation | OpenAI API or local Hugging Face causal LM |
| API | FastAPI + Uvicorn |
| Data | Hugging Face Datasets, JSONL, AWS S3 (boto3) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY (and optional AWS_* / callback secret)
```

### 1. Prepare the Swiss legal corpus (local)

```bash
python swiss_legal_dataset.py --max-train 5000 --max-val 500
```

Or point at S3 instead of local files:

```bash
export AWS_S3_DATA_BUCKET=your-bucket
export AWS_S3_DATA_PREFIX=swiss_legal/
```

### 2. Run a prediction (CLI)

```bash
python legal_prediction.py --case-context "Claimant argues … Defendant responds …"
```

### 3. Serve the API

```bash
uvicorn api:app --host 0.0.0.0 --port 3001
```

```bash
curl -X POST http://localhost:3001/api/legal-predict \
  -H 'Content-Type: application/json' \
  -d '{"case_context":"…","top_k":5}'
# → {"job_id":"…"}  then GET /api/legal-predict/jobs/{job_id}
```

### 4. Evaluate on gold cases

```bash
python eval_gold.py --max 20 --output results.jsonl
```

## Configuration

All secrets and deploy knobs come from the environment (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | LLM generation (when not using `--model-path`) |
| `LEGAL_DATA_DIR` | Local corpus path (default `data/swiss_legal`) |
| `AWS_S3_DATA_BUCKET` / `AWS_S3_DATA_PREFIX` | Stream or sync corpus from S3 |
| `AWS_REGION` | S3 client region |
| `LEGAL_PREDICTION_CALLBACK_SECRET` | Shared secret for webhook callbacks |
| `LEGAL_FILTER_PUBLIC_AUTHORITY_PARTIES` | Drop public-authority party cases from retrieval (default on) |
| `PORT` | API port (default `3001`) |

## Repo layout

```
api.py                              # FastAPI service (async jobs + callbacks)
legal_prediction.py                 # RAG retrieval + LLM prediction/reasoning
swiss_legal_dataset.py              # HF dataset → JSONL export
s3_data.py                          # S3 sync / stream helpers
precompute_embeddings.py            # Corpus embedding cache (.npy)
precompute_embeddings_from_labels.py
eval_gold.py                        # Held-out evaluation
requirements.txt
```

## Notes

- Do not commit `.env`, local `data/`, `*.jsonl`, or `*.npy` — they are gitignored.
- The gold split must stay out of the retrieval corpus so eval remains honest.
- Multilingual embeddings matter here: Swiss decisions are DE/FR/IT-heavy.

## License

Private / all rights reserved unless otherwise stated by the author.
