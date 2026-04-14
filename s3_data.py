"""
Sync Swiss legal data from S3 to a local directory so the RAG pipeline can read JSONL files from disk.

Set in environment:
  AWS_S3_DATA_BUCKET   – S3 bucket name (e.g. my-app-legal-data)
  AWS_S3_DATA_PREFIX   – Optional prefix under the bucket (e.g. swiss_legal/ or data/swiss_legal/)
  AWS_REGION           – Optional; default region is used if unset

Objects under the prefix are downloaded into local_dir, preserving relative keys as file names.
Example: s3://bucket/swiss_legal/val.jsonl -> local_dir/val.jsonl
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def ensure_data_from_s3(local_dir: Path) -> None:
    """
    If AWS_S3_DATA_BUCKET is set, download objects from the given prefix into local_dir.
    If not set, do nothing (local data/ is used as-is).
    """
    bucket = os.environ.get("AWS_S3_DATA_BUCKET", "").strip()
    if not bucket:
        log.debug("AWS_S3_DATA_BUCKET not set; using local data only")
        return

    prefix = (os.environ.get("AWS_S3_DATA_PREFIX") or "swiss_legal/").strip()
    if not prefix.endswith("/"):
        prefix += "/"

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed; cannot sync data from S3. pip install boto3")
        return

    region = os.environ.get("AWS_REGION")
    client = boto3.client("s3", region_name=region if region else None)
    paginator = client.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            name = Path(rel).name if rel else key.split("/")[-1]
            if not name:
                continue
            dest = local_dir / name
            try:
                client.download_file(bucket, key, str(dest))
                count += 1
                log.info("Downloaded s3://%s/%s -> %s", bucket, key, dest)
            except Exception as e:
                log.exception("Failed to download s3://%s/%s: %s", bucket, key, e)

    if count:
        log.info("S3 sync complete: %d file(s) under %s", count, local_dir)
    else:
        log.warning("S3 sync: no objects found under s3://%s/%s", bucket, prefix)


def load_jsonl_from_s3(bucket: str, prefix: str, filename: str, max_lines: int = 0):
    """
    Stream a JSONL file from S3 and yield parsed JSON objects (one per line).
    If max_lines > 0, stop after that many lines.
    """
    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed; cannot load from S3")
        return
    key = (prefix.rstrip("/") + "/" + filename).lstrip("/")
    region = os.environ.get("AWS_REGION")
    client = boto3.client("s3", region_name=region if region else None)
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        err_code = getattr(e, "response", None) and e.response.get("Error", {}).get("Code")
        if err_code == "NoSuchKey":
            log.debug("S3 key not found: s3://%s/%s", bucket, key)
        else:
            log.exception("Failed to get s3://%s/%s: %s", bucket, key, e)
        return
    import json
    body = resp["Body"]
    count = 0
    for line in body.iter_lines():
        if line is None or (max_lines and count >= max_lines):
            break
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
            count += 1
        except json.JSONDecodeError:
            continue
