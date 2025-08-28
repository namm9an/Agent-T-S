import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

JOBS_DIR = Path(os.getenv("JOBS_DIR", "./jobs")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./uploads")).resolve()

# Ensure directories exist at import time
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ISOFORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISOFORMAT)


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def create_job(job_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "inputs": inputs,
        "result": {"transcript": None, "summary": None},
        "error": None,
        "timestamps": {"created_at": now_iso(), "updated_at": now_iso()},
    }
    save_json_atomic(job_path(job_id), job)
    return job


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = job_path(job_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    job = load_job(job_id)
    if not job:
        return None
    # Merge shallow updates
    for k, v in updates.items():
        if k == "result" and isinstance(v, dict):
            # Merge into existing result
            job.setdefault("result", {})
            for rk, rv in v.items():
                job["result"][rk] = rv
        else:
            job[k] = v
    job["timestamps"]["updated_at"] = now_iso()
    save_json_atomic(job_path(job_id), job)
    return job


def set_status(job_id: str, status: str) -> Optional[Dict[str, Any]]:
    return update_job(job_id, status=status)


def set_error(job_id: str, error: Any) -> Optional[Dict[str, Any]]:
    return update_job(job_id, status="failed", error=error)


def set_result(job_id: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # result may include transcript and/or summary
    return update_job(job_id, result=result, status="completed")


def allocate_upload_path(original_filename: str) -> Path:
    ext = Path(original_filename).suffix or ""
    fname = f"{uuid.uuid4()}{ext}"
    return UPLOADS_DIR / fname
