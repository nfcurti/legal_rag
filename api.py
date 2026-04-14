#!/usr/bin/env python3
"""
Legal prediction API for the Next.js frontend. Run with:
  uvicorn api:app --host 0.0.0.0 --port 3001

POST /api/legal-predict with body: { "case_context": "...", "callback_url"?: "...", "case_id"?: "..." }
  Returns 202 Accepted with { "job_id": "..." }. When job completes, if callback_url is set, POSTs result there.
GET /api/legal-predict/jobs/{job_id}
  Returns { "status": "pending"|"running"|"completed"|"failed", "result"?: {...}, "error"?: str }
"""
import json
import logging
import os
import threading
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env-style file into os.environ (non-destructive)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"").strip()
            if k and v:
                os.environ.setdefault(k, v)


# Prefer values already present in the process env; fall back to .env then .env.local.
_load_env_file(ROOT / ".env")
_load_env_file(ROOT / ".env.local")

# When AWS_S3_DATA_BUCKET is set, we load corpus from S3 on first use (no local download).
# When not set, optionally sync from S3 into local dir, or use existing local data.
_data_dir = Path(os.environ.get("LEGAL_DATA_DIR", str(ROOT / "data" / "swiss_legal")))
from s3_data import ensure_data_from_s3

_use_s3_stream = bool(os.environ.get("AWS_S3_DATA_BUCKET", "").strip())
if _use_s3_stream:
    log.info("Legal data source: S3 stream (bucket=%s, prefix=%s) — no local download", os.environ.get("AWS_S3_DATA_BUCKET"), os.environ.get("AWS_S3_DATA_PREFIX", "swiss_legal/"))
else:
    ensure_data_from_s3(_data_dir)
    log.info("Legal data source: local (LEGAL_DATA_DIR=%s)", _data_dir)

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _warm_up():
    """Load embedding model and corpus on startup so first request is fast.
    Uses same max_cases as production (5000) so precomputed .npy can be used."""
    from legal_prediction import retrieve_similar_cases
    retrieve_similar_cases("warm up", top_k=1, max_cases=5000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up: loading embedding model and corpus (may take 1–3 min)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warm_up)
    log.info("Warm-up complete. Ready to serve.")
    yield


app = FastAPI(title="Legal prediction RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store: job_id -> { status, result?, error? }
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Keep last 500 jobs; drop oldest when full
_MAX_JOBS = 500


class PredictRequest(BaseModel):
    case_context: str
    top_k: int = 5
    case_category: Optional[str] = None
    case_subcategory: Optional[str] = None
    callback_url: Optional[str] = None
    case_id: Optional[str] = None


def _fire_callback(callback_url: str, case_id: Optional[str], job_id: str, status: str, result: Any = None, error: Optional[str] = None) -> None:
    """POST job result to Next.js callback URL. Runs in background thread; does not block."""
    secret = os.environ.get("LEGAL_PREDICTION_CALLBACK_SECRET", "").strip()
    body = json.dumps({
        "job_id": job_id,
        "case_id": case_id,
        "status": status,
        "result": result,
        "error": error,
    }).encode("utf-8")
    req = urllib.request.Request(
        callback_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Callback-Secret": secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.getcode()
            if 200 <= code < 300:
                log.info("[job %s] Callback to %s succeeded", job_id[:8], callback_url[:50])
            else:
                log.warning("[job %s] Callback returned %s", job_id[:8], code)
    except urllib.error.HTTPError as e:
        log.warning("[job %s] Callback HTTP error: %s %s", job_id[:8], e.code, e.reason)
    except Exception as e:
        log.exception("[job %s] Callback failed: %s", job_id[:8], e)


def _run_prediction(
    job_id: str,
    case_context: str,
    top_k: int,
    case_category: Optional[str] = None,
    case_subcategory: Optional[str] = None,
    callback_url: Optional[str] = None,
    case_id: Optional[str] = None,
) -> None:
    """Background worker: run prediction, store result, and optionally POST to callback_url."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        from legal_prediction import predict_with_reasoning

        log.info("[job %s] Calling predict_with_reasoning (top_k=%s)...", job_id[:8], top_k)
        out = predict_with_reasoning(
            case_context,
            top_k=top_k,
            case_category=case_category,
            case_subcategory=case_subcategory,
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        precs = out.get("precedent_cases")
        if precs:
            out["precedent_cases"] = [_format_precedent(c) for c in precs]
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = out
        log.info("[job %s] Completed (prediction=%s)", job_id[:8], bool(out.get("prediction")))
        if callback_url:
            _fire_callback(callback_url, case_id, job_id, "completed", result=out, error=None)
    except Exception as e:
        log.exception("[job %s] predict_with_reasoning failed: %s", job_id[:8], e)
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
        if callback_url:
            _fire_callback(callback_url, case_id, job_id, "failed", result=None, error=str(e))


@app.post("/api/legal-predict")
def legal_predict(req: PredictRequest):
    """Start prediction job. Returns 202 with job_id; poll GET /api/legal-predict/jobs/{job_id} for result."""
    log.info("POST /api/legal-predict received (top_k=%s)", req.top_k)
    case_context = (req.case_context or "").strip()
    if not case_context:
        log.warning("Rejected: missing case_context")
        raise HTTPException(status_code=400, detail="Provide 'case_context' (summary of the case)")
    log.info("case_context length=%d chars, preview: %s", len(case_context), case_context[:200].replace("\n", " ") + ("..." if len(case_context) > 200 else ""))

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        while len(_jobs) >= _MAX_JOBS:
            # Drop oldest job (arbitrary: first key)
            old = next(iter(_jobs))
            del _jobs[old]
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}

    callback_url = (req.callback_url or "").strip() or None
    case_id = (req.case_id or "").strip() or None
    if callback_url and not case_id:
        log.warning("callback_url provided without case_id; callback will send case_id=null")

    thread = threading.Thread(
        target=_run_prediction,
        args=(job_id, case_context, req.top_k, req.case_category, req.case_subcategory),
        kwargs={"callback_url": callback_url, "case_id": case_id},
        daemon=True,
    )
    thread.start()
    return JSONResponse(content={"job_id": job_id}, status_code=202)


@app.get("/api/legal-predict/jobs/{job_id}")
def get_job(job_id: str):
    """Return job status and result (if completed) or error (if failed)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _format_precedent(c: dict) -> dict:
    """Format one precedent: doctrine (court's reasoning), relevance to the case, and full fields."""
    raw_lang = c.get("language")
    lang = (raw_lang.strip().lower()[:2] if isinstance(raw_lang, str) and raw_lang.strip() else "") or "de"
    doctrine = c.get("considerations") or ""
    return {
        "case_id": c.get("decision_id", ""),
        "decision_id": c.get("decision_id", ""),
        "chamber": c.get("chamber"),
        "law_area": c.get("law_area"),
        # Optional labels used by the arbitrator UI.
        "category": c.get("category") or c.get("caseCategory") or c.get("case_category") or c.get("ai_case_category"),
        "subcategory": c.get("subcategory") or c.get("caseSubcategory") or c.get("case_subcategory") or c.get("ai_case_subcategory"),
        "year": c.get("year"),
        "language": lang,
        "facts_excerpt": (c.get("facts") or "")[:400],
        "rulings_excerpt": (c.get("rulings") or "")[:300],
        "facts": c.get("facts") or "",
        "rulings": c.get("rulings") or "",
        "considerations": doctrine,
        "doctrine": doctrine,
        "doctrine_excerpt": doctrine[:500] if doctrine else "",
        "relevance": c.get("relevance"),
    }


@app.get("/api/health")
def health():
    log.debug("GET /api/health")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3001))
    uvicorn.run(app, host="0.0.0.0", port=port)
