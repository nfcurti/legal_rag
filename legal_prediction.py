#!/usr/bin/env python3
"""
Legal prediction with reasoning from case context (claim/defense content).

Input: case context as text (e.g. claim and defense content)—not documents per se,
but their content passed as context in the prompt.
Output: prediction (likely outcome) + legal reasoning.

Uses the Swiss legal dataset (data/swiss_legal/) for retrieval: similar precedent
cases are found via embedding-based RAG (sentence-transformers, default
paraphrase-multilingual-MiniLM-L12-v2) or TF-IDF, and added to the prompt so
prediction and reasoning are grounded in real Federal Supreme Court decisions.
"""

import json
import os
from pathlib import Path
from typing import Callable, Optional

# Path to Swiss legal dataset (override with LEGAL_DATA_DIR when using S3 or custom deploy)
_default_data_dir = Path(__file__).resolve().parent / "data" / "swiss_legal"
DATA_DIR = Path(os.environ.get("LEGAL_DATA_DIR", _default_data_dir))

# Module-level index (built once, reused)
_dataset_cache: Optional[list[dict]] = None
_tfidf_cache: object = None
_tfidf_matrix_cache: object = None
_embedding_model_cache: Optional[tuple[str, object]] = None  # (model_name, model)
_corpus_embeddings_cache: Optional[tuple[tuple[int, str], object]] = None  # ((corpus_id, model_name), embeddings)
# For precomputed corpus embeddings stored as .npy on disk (local deployments only)
_precomputed_embeddings_cache: Optional[tuple[tuple[int, str], object]] = None  # ((n_cases, path), embeddings)


def _looks_like_public_body(text: str) -> bool:
    """Heuristic: True if the case appears to involve a public authority as party.

    We only inspect the header/intro (first ~800 chars) where parties are listed.
    This is language-agnostic-ish and relies on common tokens like 'Bundesamt',
    'Kanton', 'Gemeinde', etc. It is intentionally conservative: when in doubt,
    we keep the case (return False).
    """
    head = (text or "").strip().lower()[:800]
    if not head:
        return False
    markers = [
        "schweizerische eidgenossenschaft",
        "gegen Schweizerische Eidgenossenschaft ",
        "bundeskanzlei",
        "kanton ",
        "kantons ",
        "kantonalen",
        "kantonale ",
        "gemeinde ",
        "stadt ",
        "steuerverwaltung",
        "sozialversicherungsanstalt",
        "ausgleichskasse",
        "iv-stelle",
        "ahv-ausgleichskasse",
        "suva ",
    ]
    return any(m in head for m in markers)

def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _record_to_case(rec: dict) -> Optional[dict]:
    """Convert one JSONL record to corpus case dict, or None if filtered out.

    For similarity search we primarily use `url_text` (excerpt of the case)
    when available; otherwise we fall back to `facts`. The retrieval code
    downstream always reads from the unified `facts` field.
    """
    # Prefer url_text (excerpt) as the text to compare against the case summary.
    text = (rec.get("url_text") or rec.get("facts") or "").strip()
    if not text or len(text) < 100:
        return None
    # Drop cases where one of the parties is a public authority (federal/cantonal).
    if _env_flag("LEGAL_FILTER_PUBLIC_AUTHORITY_PARTIES", True) and _looks_like_public_body(text):
        return None
    if rec.get("law_area") == "penal_law":
        return None
    return {
        "decision_id": rec.get("decision_id", ""),
        "chamber": rec.get("chamber"),
        # Store the text used for retrieval under 'facts' so the rest of the
        # pipeline (retrieval, formatting) continues to work unchanged.
        "facts": text,
        "considerations": (rec.get("considerations") or "").strip(),
        "rulings": (rec.get("rulings") or "").strip(),
        "law_area": rec.get("law_area"),
        "year": rec.get("year"),
        "language": (rec.get("language") or "de").lower()[:2],
        # Optional labels used by the arbitrator UI.
        # Prefer explicit labels from the dataset; fall back to law_area/law_sub_area.
        "category": _clean_optional_label(
            rec.get("category")
            or rec.get("case_category")
            or rec.get("ai_case_category")
            or rec.get("law_area")
        ),
        "subcategory": _clean_optional_label(
            rec.get("subcategory")
            or rec.get("case_subcategory")
            or rec.get("ai_case_subcategory")
            or rec.get("law_sub_area")
        ),
    }


def _clean_optional_label(v: object) -> Optional[str]:
    """Normalize optional string labels (treat 'nan'/empty as missing)."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.lower() == "nan":
            return None
        return s
    return str(v).strip() or None


def _load_swiss_legal_corpus(split: str = "val", max_cases: int = 20_000) -> list[dict]:
    """Load Swiss legal cases from JSONL for retrieval. Uses val by default (smaller, diverse).
    Does not load gold.jsonl — keep gold as a held-out evaluation set, not in RAG/training.
    When AWS_S3_DATA_BUCKET is set, loads directly from S3 (no local download).

    If LEGAL_LABELS_JSONL_PATH is set, load from that JSONL file instead of S3/local,
    using the same _record_to_case helper (e.g. for custom curated datasets)."""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    labels_path = os.environ.get("LEGAL_LABELS_JSONL_PATH", "").strip()
    if labels_path:
        path_obj = Path(labels_path)
        if path_obj.exists():
            cases: list[dict] = []
            with path_obj.open(encoding="utf-8") as f:
                for line in f:
                    if len(cases) >= max_cases:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    case = _record_to_case(rec)
                    if case:
                        cases.append(case)
            _dataset_cache = cases
            return cases

    bucket = os.environ.get("AWS_S3_DATA_BUCKET", "").strip()
    prefix = (os.environ.get("AWS_S3_DATA_PREFIX") or "swiss_legal/").strip()
    if not prefix.endswith("/"):
        prefix += "/"

    if bucket:
        # Stream from S3 (no local data dir on EC2)
        from s3_data import load_jsonl_from_s3
        for filename in (f"{split}.jsonl", "train.jsonl"):
            cases = []
            for rec in load_jsonl_from_s3(bucket, prefix, filename, max_lines=0):
                if len(cases) >= max_cases:
                    break
                case = _record_to_case(rec)
                if case:
                    cases.append(case)
            if cases:
                _dataset_cache = cases
                return cases
        _dataset_cache = []
        return []

    # Local files
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        path = DATA_DIR / "train.jsonl"
    if not path.exists():
        _dataset_cache = []
        return []
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(cases) >= max_cases:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = _record_to_case(rec)
            if case:
                cases.append(case)
    _dataset_cache = cases
    return cases


def _is_bge_m3_model(embedding_model: str) -> bool:
    """True if the model ID is a BGE-M3 variant (use FlagEmbedding)."""
    return "bge-m3" in embedding_model.lower()


def _select_progressive_indices(
    corpus: list[dict],
    sims: object,
    top_k: int,
    *,
    case_subcategory: Optional[str],
    case_category: Optional[str],
) -> list[int]:
    """
    Select indices in a progressive backoff order:
    1) same subcategory
    2) same category
    3) global fallback (any remaining)
    """
    import numpy as np

    if top_k <= 0:
        return []

    sims_np = np.asarray(sims, dtype=np.float32).ravel()
    n = len(corpus)
    if sims_np.shape[0] != n:
        # Defensive: fall back to global top_k if shapes mismatch.
        ranked = np.argsort(sims_np)[-top_k:][::-1]
        return ranked.astype(int).tolist()

    def top_for_mask(mask: "np.ndarray", k: int) -> "np.ndarray":
        idx = np.nonzero(mask)[0]
        if idx.size == 0 or k <= 0:
            return np.array([], dtype=int)
        local_sims = sims_np[idx]
        if idx.size <= k:
            order = np.argsort(local_sims)[::-1]
        else:
            order = np.argsort(local_sims)[-k:][::-1]
        return idx[order].astype(int)

    selected: list[int] = []
    seen = np.zeros(n, dtype=bool)

    # Step 1: subcategory
    if case_subcategory:
        mask_sub = np.array([c.get("subcategory") == case_subcategory for c in corpus], dtype=bool)
        picked = top_for_mask(mask_sub, top_k)
        selected.extend(picked.tolist())
        seen[picked] = True

    # Step 2: category
    if len(selected) < top_k and case_category:
        remaining = top_k - len(selected)
        mask_cat = np.array([c.get("category") == case_category for c in corpus], dtype=bool) & ~seen
        picked = top_for_mask(mask_cat, remaining)
        selected.extend(picked.tolist())
        seen[picked] = True

    # Step 4: global fallback (fill whatever remains)
    if len(selected) < top_k:
        remaining = top_k - len(selected)
        mask_any = ~seen
        picked = top_for_mask(mask_any, remaining)
        selected.extend(picked.tolist())

    return selected


def _retrieve_with_bge_m3(
    query: str,
    corpus: list[dict],
    top_k: int,
    embedding_model: str,
    batch_size: int = 64,
    max_length: int = 8192,
    case_subcategory: Optional[str] = None,
    case_category: Optional[str] = None,
) -> list[dict]:
    """Retrieve top_k cases using BGE-M3 dense embeddings (FlagEmbedding). Caches model and corpus."""
    global _embedding_model_cache, _corpus_embeddings_cache
    try:
        from FlagEmbedding import BGEM3FlagModel
        import numpy as np
    except ImportError:
        raise ImportError(
            "BGE-M3 retrieval requires FlagEmbedding. Install with: pip install FlagEmbedding"
        ) from None

    # Normalize model name (allow "bge-m3" as shorthand)
    model_name = embedding_model if embedding_model else "BAAI/bge-m3"
    if model_name.lower() == "bge-m3":
        model_name = "BAAI/bge-m3"

    # Load model (cached)
    if _embedding_model_cache is None or _embedding_model_cache[0] != model_name:
        _embedding_model_cache = (model_name, BGEM3FlagModel(model_name, use_fp16=True))
    model = _embedding_model_cache[1]

    texts = [c["facts"] for c in corpus]
    corpus_id = id(corpus)
    cache_key = (corpus_id, model_name)

    # Embed corpus (cached)
    if _corpus_embeddings_cache is None or _corpus_embeddings_cache[0] != cache_key:
        out = model.encode(texts, batch_size=batch_size, max_length=max_length)
        dense = out["dense_vecs"]
        _corpus_embeddings_cache = (cache_key, np.asarray(dense, dtype=np.float32))
    corpus_emb = _corpus_embeddings_cache[1]

    # Embed query
    q_out = model.encode([query], max_length=max_length)
    q_emb = np.asarray(q_out["dense_vecs"], dtype=np.float32)

    # Cosine similarity
    norms = np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    corpus_n = corpus_emb / norms
    q_n = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) or 1.0)
    sims = (corpus_n @ q_n.T).ravel()

    top_indices = _select_progressive_indices(
        corpus,
        sims,
        top_k,
        case_subcategory=case_subcategory,
        case_category=case_category,
    )

    out = []
    for i in top_indices:
        c = corpus[i]
        score = float(sims[i])
        out.append({
            "decision_id": c.get("decision_id", ""),
            "chamber": c.get("chamber"),
            "facts": c["facts"],
            "considerations": c["considerations"] or "(no considerations)",
            "rulings": c["rulings"] or "(no ruling)",
            "law_area": c.get("law_area"),
            "year": c.get("year"),
            "language": c.get("language"),
            "category": c.get("category"),
            "subcategory": c.get("subcategory"),
            "relevance": round(score, 4),
        })
    return out


def _retrieve_with_embeddings(
    query: str,
    corpus: list[dict],
    top_k: int,
    embedding_model: str,
    batch_size: int = 64,
    case_subcategory: Optional[str] = None,
    case_category: Optional[str] = None,
) -> list[dict]:
    """Retrieve top_k cases by embedding cosine similarity (sentence-transformers).

    If LEGAL_EMBEDDINGS_PATH (or default embeddings_*.npy) exists and AWS_S3_DATA_BUCKET
    is not set, uses precomputed corpus embeddings from .npy to avoid embedding the
    corpus at runtime. Otherwise, falls back to computing corpus embeddings on the fly.
    """
    global _embedding_model_cache, _corpus_embeddings_cache, _precomputed_embeddings_cache
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError(
            "Embedding-based RAG requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from None

    # Load model (cached) – always needed to embed the query
    if _embedding_model_cache is None or _embedding_model_cache[0] != embedding_model:
        _embedding_model_cache = (embedding_model, SentenceTransformer(embedding_model))
    model = _embedding_model_cache[1]

    texts = [c["facts"] for c in corpus]

    corpus_emb = None

    # 1) Try to use precomputed embeddings from .npy (works for both local and S3 corpus)
    split_name = "val"
    default_path = DATA_DIR / f"embeddings_{split_name}.npy"
    embeddings_path = os.environ.get("LEGAL_EMBEDDINGS_PATH", str(default_path))
    key = (len(corpus), embeddings_path)
    try:
        if _precomputed_embeddings_cache is None or _precomputed_embeddings_cache[0] != key:
            path_obj = Path(embeddings_path)
            if not path_obj.is_absolute():
                path_obj = Path(__file__).resolve().parent / path_obj
            if path_obj.exists():
                emb = np.load(path_obj, allow_pickle=True)
                if emb.shape[0] != len(corpus):
                    # Mismatch between corpus size and embeddings; log and ignore precomputed file
                    print(
                        f"[legal_prediction] Precomputed embeddings at {path_obj} have "
                        f"{emb.shape[0]} rows, expected {len(corpus)}; ignoring."
                    )
                else:
                    print(
                        f"[legal_prediction] Using precomputed embeddings from {path_obj} "
                        f"with shape {emb.shape}"
                    )
                    _precomputed_embeddings_cache = (key, np.asarray(emb, dtype=np.float32))
            # If file does not exist, fall back to on-the-fly encoding below
        if _precomputed_embeddings_cache is not None and _precomputed_embeddings_cache[0] == key:
            corpus_emb = _precomputed_embeddings_cache[1]
    except Exception as e:  # pragma: no cover - defensive; falls back to dynamic encoding
        print(f"[legal_prediction] Failed to load precomputed embeddings from {embeddings_path}: {e}")
        _precomputed_embeddings_cache = None

    # 2) Fallback: embed corpus on the fly
    if corpus_emb is None:
        _path = Path(embeddings_path)
        if not _path.is_absolute():
            _path = Path(__file__).resolve().parent / _path
        if not _path.exists():
            print(f"[legal_prediction] No precomputed embeddings at {_path}; embedding corpus on the fly.")
        corpus_id = id(corpus)
        cache_key = (corpus_id, embedding_model)
        if _corpus_embeddings_cache is None or _corpus_embeddings_cache[0] != cache_key:
            emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            _corpus_embeddings_cache = (cache_key, np.asarray(emb, dtype=np.float32))
        corpus_emb = _corpus_embeddings_cache[1]

    # Embed query
    q_emb = model.encode([query], batch_size=1)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    # Cosine similarity
    norms = np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    corpus_n = corpus_emb / norms
    q_n = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) or 1.0)
    sims = (corpus_n @ q_n.T).ravel()

    top_indices = _select_progressive_indices(
        corpus,
        sims,
        top_k,
        case_subcategory=case_subcategory,
        case_category=case_category,
    )

    out = []
    for i in top_indices:
        c = corpus[i]
        score = float(sims[i])
        out.append({
            "decision_id": c.get("decision_id", ""),
            "chamber": c.get("chamber"),
            "facts": c["facts"],
            "considerations": c["considerations"] or "(no considerations)",
            "rulings": c["rulings"] or "(no ruling)",
            "law_area": c.get("law_area"),
            "year": c.get("year"),
            "language": c.get("language"),
            "category": c.get("category"),
            "subcategory": c.get("subcategory"),
            "relevance": round(score, 4),
        })
    return out


def retrieve_similar_cases(
    case_context: str,
    *,
    top_k: int = 5,
    split: str = "val",
    max_cases: int = 5_000,
    use_embeddings: bool = True,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    case_subcategory: Optional[str] = None,
    case_category: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the Swiss legal dataset the cases most similar to the case context.

    case_context: summary of the case (claim, defense, rebuttals, exhibits, etc.).
    By default uses sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2).
    Pass a BGE-M3 model ID (e.g. BAAI/bge-m3) to use FlagEmbedding instead. Set
    use_embeddings=False for TF-IDF retrieval. Returns top_k cases with full facts,
    considerations, rulings.
    """
    corpus = _load_swiss_legal_corpus(split=split, max_cases=max_cases)
    if not corpus:
        return []

    query = case_context.strip()

    if use_embeddings:
        if _is_bge_m3_model(embedding_model):
            return _retrieve_with_bge_m3(
                query,
                corpus,
                top_k,
                embedding_model,
                case_subcategory=case_subcategory,
                case_category=case_category,
            )
        return _retrieve_with_embeddings(
            query,
            corpus,
            top_k,
            embedding_model,
            case_subcategory=case_subcategory,
            case_category=case_category,
        )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    global _tfidf_cache, _tfidf_matrix_cache

    texts = [c["facts"] for c in corpus]
    if _tfidf_cache is None:
        vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)
        _tfidf_matrix_cache = vectorizer.fit_transform(texts)
        _tfidf_cache = vectorizer
    else:
        vectorizer = _tfidf_cache

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix_cache).ravel()

    top_indices = _select_progressive_indices(
        corpus,
        sims,
        top_k,
        case_subcategory=case_subcategory,
        case_category=case_category,
    )

    out = []
    for i in top_indices:
        c = corpus[i]
        score = float(sims[i])
        out.append({
            "decision_id": c.get("decision_id", ""),
            "chamber": c.get("chamber"),
            "facts": c["facts"],
            "considerations": c["considerations"] or "(no considerations)",
            "rulings": c["rulings"] or "(no ruling)",
            "law_area": c.get("law_area"),
            "year": c.get("year"),
            "language": c.get("language"),
            "category": c.get("category"),
            "subcategory": c.get("subcategory"),
            "relevance": round(score, 4),
        })
    return out


def build_prompt(
    case_context: str,
    *,
    precedent_cases: Optional[list[dict]] = None,
    top_k: int = 5,
    use_dataset: bool = True,
    use_embeddings: bool = True,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    max_cases: int = 5_000,
    jurisdiction: str = "Swiss law",
) -> tuple[str, str]:
    """
    Build system and user messages for legal prediction with reasoning.

    case_context: summary of the case (claim, defense, rebuttals, exhibits, etc.).
    If use_dataset is True (default), retrieves top_k similar cases from the Swiss
    legal dataset and includes them as precedent. By default uses sentence-transformers;
    set use_embeddings=False for TF-IDF. You can also pass precedent_cases directly.
    """
    if use_dataset and (precedent_cases is None or len(precedent_cases) == 0):
        precedent_cases = retrieve_similar_cases(
            case_context,
            top_k=top_k,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            max_cases=max_cases,
        )

    system = (
        "You are a legal analyst applying Swiss law and Swiss Federal Supreme Court practice. "
        "You must (1) predict the likely outcome (who is likely to prevail and on which points), "
        "(2) provide a comprehensive legal basis explaining the applicable articles and how they relate to this specific case, "
        "and (3) provide precedents reasoning based only on the precedent cases listed (Precedent 1 through 5) to justify the suggested ruling."
    )

    parts = []
    # Use only the first 5 precedents: these are the ones we show to the user and the only ones to reason on.
    display_precedents = (precedent_cases or [])[:5]

    if display_precedents:
        parts.append(
            "Relevant precedent cases from the Swiss Federal Supreme Court dataset "
            "(in Precedents reasoning you must refer ONLY to these precedent cases below—Precedent 1 through 5—and to no others):\n"
        )
        for i, c in enumerate(display_precedents, 1):
            area_yr = []
            if c.get("law_area"):
                area_yr.append(str(c["law_area"]))
            if c.get("year"):
                area_yr.append(str(c["year"]))
            meta = f" [{', '.join(area_yr)}]" if area_yr else ""
            parts.append(
                f"--- Precedent {i}{meta} ---\n"
                f"Facts:\n{c['facts']}\n\n"
                f"Court's reasoning:\n{c['considerations']}\n\n"
                f"Outcome:\n{c['rulings']}\n"
            )
        parts.append("--- Case to analyze ---\n")

    context = case_context.strip()
    parts.append(
        "Case context (summary of claim, defense, rebuttals, exhibits):\n"
        f"{context}\n\n"
        "Provide your answer in this exact format:\n"
        "Prediction: [one or two sentences: likely outcome and who prevails on what]\n"
        "Legal basis: [comprehensive text that explains the applicable articles (e.g. Art. 93 OR, Art. 97 CO) and how they relate to this specific case. Cite provisions clearly, then explain how each applies to the parties' positions and the dispute. Do not discuss precedent cases here.]\n"
        "Precedents reasoning: [reasoning that uses ONLY the precedent cases listed above (Precedent 1 through 5) to justify the suggested ruling. Do not cite or reason about any other precedents. Explain how the court's reasoning or outcome in these precedents supports your prediction.]"
    )

    return system, "\n".join(parts)


def call_llm_openai(system: str, user: str, *, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    """Call OpenAI (or compatible) API. Set OPENAI_API_KEY or pass api_key."""
    try:
        from openai import OpenAI
        import httpx
    except ImportError as e:
        raise ImportError("Install openai and httpx: pip install openai httpx") from e

    # Use explicit httpx client to avoid openai/httpx version mismatch on 'proxies' argument
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        http_client=httpx.Client(),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# Cache for local model (avoid reloading every call)
_local_model_cache: Optional[tuple[str, object, object]] = None


def call_llm_local(
    system: str,
    user: str,
    *,
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Call a local Hugging Face model (e.g. fine-tuned LoRA adapter).

    model_path: path to the saved adapter (output of finetune_legal.py) or base model.
    """
    global _local_model_cache
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("Install transformers and peft to use local model") from e

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    cache_key = str(path.resolve())
    if _local_model_cache is not None and _local_model_cache[0] == cache_key:
        model, tokenizer = _local_model_cache[1], _local_model_cache[2]
    else:
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_name = adapter.get("base_model_name_or_path", cache_key)
            tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                trust_remote_code=True,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, cache_key)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cache_key, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                cache_key,
                trust_remote_code=True,
                device_map="auto",
            )
        _local_model_cache = (cache_key, model, tokenizer)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


def parse_prediction_and_reasoning(response: str) -> dict:
    """
    Parse model output into prediction, legal_basis, and precedents_reasoning.
    Expects "Prediction:", "Legal basis:", "Precedents reasoning:" (or legacy "Reasoning:").
    """
    out = {
        "prediction": "",
        "reasoning": "",
        "legal_basis": "",
        "precedents_reasoning": "",
        "raw": response,
    }
    current = None
    current_key = None
    lines = response.split("\n")

    def flush():
        if current_key and current:
            text = "\n".join(current).strip()
            if current_key == "prediction":
                out["prediction"] = text
            elif current_key == "legal_basis":
                out["legal_basis"] = text
                out["reasoning"] = text  # backward compat
            elif current_key == "reasoning":
                out["reasoning"] = text
                if not out["legal_basis"]:
                    out["legal_basis"] = text
            elif current_key == "precedents_reasoning":
                out["precedents_reasoning"] = text

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("prediction:"):
            flush()
            current_key = "prediction"
            current = [stripped[10:].strip().lstrip(": ")]
        elif lower.startswith("legal basis:"):
            flush()
            current_key = "legal_basis"
            current = [stripped[12:].strip().lstrip(": ")]
        elif lower.startswith("precedents reasoning:"):
            flush()
            current_key = "precedents_reasoning"
            current = [stripped[20:].strip().lstrip(": ")]
        elif lower.startswith("reasoning:"):
            flush()
            current_key = "reasoning"
            current = [stripped[10:].strip().lstrip(": ")]
        elif current_key is not None and (stripped or current):
            if current is not None:
                current.append(line)
    flush()

    # Legacy: if no structured headers, first para = prediction, rest = reasoning/legal_basis
    if not out["prediction"] and not out["legal_basis"] and response:
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if paragraphs:
            out["prediction"] = paragraphs[0]
            rest = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
            out["reasoning"] = rest
            out["legal_basis"] = rest
    return out


def predict_with_reasoning(
    case_context: str,
    *,
    top_k: int = 5,
    use_dataset: bool = True,
    use_embeddings: bool = True,
    embedding_model: str = "BAAI/bge-m3",
    max_cases: int = 5_000,
    case_subcategory: Optional[str] = None,
    case_category: Optional[str] = None,
    jurisdiction: str = "Swiss law",
    llm_call: Optional[Callable[[str, str], str]] = None,
    model: str = "gpt-4o-mini",
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run legal prediction with reasoning (fine-tuning + RAG pipeline).

    case_context: summary of the case (statement of claim, defense, rebuttals, exhibits).
    - RAG: by default retrieves top_k similar cases from the Swiss legal dataset
      and passes them as precedent context to the model.
    - Fine-tuned model: pass model_path to use a local model (e.g. output of
      finetune_legal.py). Otherwise uses OpenAI with `model`.

    Returns:
        dict with keys: prediction, reasoning, raw. If use_dataset is True, also
        includes precedent_cases (list of dicts with facts, considerations, rulings,
        law_area, year) so you can inspect which cases were retrieved.

    If llm_call is provided, it is used as llm_call(system, user) -> response.
    Else if model_path is set, uses call_llm_local (fine-tuned + RAG).
    Else uses call_llm_openai.
    """
    precedent_cases: Optional[list[dict]] = None
    if use_dataset:
        precedent_cases = retrieve_similar_cases(
            case_context,
            top_k=top_k,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            max_cases=max_cases,
            case_subcategory=case_subcategory,
            case_category=case_category,
        )
    # Use first 5 for the prompt only (LLM reasons on these); return full list so swap has a backup.
    system, user = build_prompt(
        case_context,
        precedent_cases=precedent_cases,
        top_k=top_k,
        use_dataset=use_dataset,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        max_cases=max_cases,
        jurisdiction=jurisdiction,
    )
    if llm_call is not None:
        response = llm_call(system, user)
    elif model_path:
        response = call_llm_local(system, user, model_path=model_path)
    else:
        response = call_llm_openai(system, user, model=model, api_key=api_key)
    out = parse_prediction_and_reasoning(response)
    if precedent_cases is not None:
        out["precedent_cases"] = precedent_cases
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Legal prediction with reasoning (single case_context input)")
    parser.add_argument("--case-context", "--facts", type=str, dest="case_context", default=None, help="Case context: summary of claim, defense, rebuttals, exhibits (text or path)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar precedent cases from dataset (default 5)")
    parser.add_argument("--no-dataset", action="store_true", help="Do not use Swiss legal dataset for retrieval")
    parser.add_argument("--tfidf", action="store_true", help="Use TF-IDF retrieval instead of embedding-based RAG")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model for RAG (default sentence-transformers; or BAAI/bge-m3 for FlagEmbedding)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name if not using --model-path")
    parser.add_argument("--model-path", type=str, default=None, help="Path to local/fine-tuned model (enables RAG + fine-tuned)")
    parser.add_argument("--no-api", action="store_true", help="Only print the prompt, do not call API")
    args = parser.parse_args()

    def _read(s: str) -> str:
        p = Path(s)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return s

    if not args.case_context:
        parser.error("Provide --case-context (or --facts) with the case summary")
    case_context = _read(args.case_context)

    system, user = build_prompt(
        case_context,
        top_k=args.top_k,
        use_dataset=not args.no_dataset,
        use_embeddings=not args.tfidf,
        embedding_model=args.embedding_model,
    )

    if args.no_api:
        print("=== System ===\n")
        print(system)
        print("\n=== User ===\n")
        print(user)
        return

    result = predict_with_reasoning(
        case_context,
        top_k=args.top_k,
        use_dataset=not args.no_dataset,
        use_embeddings=not args.tfidf,
        embedding_model=args.embedding_model,
        llm_call=None,
        model=args.model,
        model_path=args.model_path,
    )
    print("Prediction:", result["prediction"])
    print("\nReasoning:", result["reasoning"])


if __name__ == "__main__":
    main()
