import os
import traceback
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, Header
from fastapi.responses import JSONResponse

from app.integrations.langgraph import get_client
from app.schemas import EnqueueResponse, Job, JobResponse, SummarizeRequest, LangGraphCallback
from app.services import storage
from app.utils.logger import get_logger

app = FastAPI(title="Transcription + Summarization API (LangGraph v1)")
logger = get_logger("app.main")

WAIT_TIMEOUT = float(os.getenv("WAIT_TIMEOUT_SECONDS", "180"))
POLL_INTERVAL = float(os.getenv("WAIT_POLL_INTERVAL", "1.0"))
CALLBACK_SECRET = os.getenv("LANGGRAPH_CALLBACK_SECRET")
WORKFLOW_ID = os.getenv("LANGGRAPH_WORKFLOW_ID", "transcribe_summarize_v1")


def _extract_job_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize possible LangGraph response shapes into our job.result fields.
    Expected keys: transcript (str) and summary (dict or str).
    """
    result: Dict[str, Any] = {"transcript": None, "summary": None}

    # Common shapes
    # 1) { result: { transcript: "...", summary: {...} } }
    if isinstance(data.get("result"), dict):
        r = data["result"]
        if isinstance(r.get("transcript"), str):
            result["transcript"] = r["transcript"]
        if r.get("summary") is not None:
            result["summary"] = r["summary"]

    # 2) { transcript: "...", summary: {...} }
    if result["transcript"] is None and isinstance(data.get("transcript"), str):
        result["transcript"] = data["transcript"]
    if result["summary"] is None and data.get("summary") is not None:
        result["summary"] = data["summary"]

    return result


def _complete_if_present(job_id: str, response: Dict[str, Any]) -> bool:
    result = _extract_job_result(response)
    if result.get("transcript") is not None or result.get("summary") is not None:
        storage.set_result(job_id, result)
        return True
    return False


def process_transcribe(job_id: str, file_path: str, diarize: bool = False) -> None:
    client = get_client()
    try:
        storage.set_status(job_id, "running")
        payload = {"file_path": file_path, "diarize": diarize}
        resp = client.trigger(action="transcribe", inputs=payload, job_id=job_id)

        # If immediate result returned
        if _complete_if_present(job_id, resp):
            return

        # Else, try to poll using run references from response
        run_ref: Dict[str, Any] = {}
        if "run_id" in resp:
            run_ref["run_id"] = resp["run_id"]
        run_ref["job_id"] = job_id

        data, err = client.poll_until_complete(run_ref)
        if err:
            storage.set_error(job_id, {"message": err, "upstream": resp})
            return
        if not data:
            storage.set_error(job_id, {"message": "No data received from LangGraph"})
            return
        result = _extract_job_result(data)
        storage.set_result(job_id, result)
    except Exception as e:
        logger.exception("Transcription job failed: %s", e)
        storage.set_error(job_id, {"message": str(e), "trace": traceback.format_exc()})


def process_summarize(job_id: str, source: Dict[str, Any]) -> None:
    client = get_client()
    try:
        storage.set_status(job_id, "running")
        # source contains either {"job_id": prev_id, "summary_type": "..."} or {"text": "...", "summary_type": "..."}
        resp = client.trigger(action="summarize", inputs=source, job_id=job_id)

        # If immediate result returned
        if _complete_if_present(job_id, resp):
            return

        run_ref: Dict[str, Any] = {}
        if "run_id" in resp:
            run_ref["run_id"] = resp["run_id"]
        run_ref["job_id"] = job_id

        data, err = client.poll_until_complete(run_ref)
        if err:
            storage.set_error(job_id, {"message": err, "upstream": resp})
            return
        if not data:
            storage.set_error(job_id, {"message": "No data received from LangGraph"})
            return
        result = _extract_job_result(data)
        storage.set_result(job_id, result)
    except Exception as e:
        logger.exception("Summarization job failed: %s", e)
        storage.set_error(job_id, {"message": str(e), "trace": traceback.format_exc()})


async def _wait_for_job(job_id: str, timeout: float, poll: float) -> Dict[str, Any]:
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        job = storage.load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("completed", "failed"):
            return job
        if asyncio.get_event_loop().time() >= deadline:
            return job  # timeout; return current state
        await asyncio.sleep(poll)


@app.get("/health")
async def health():
    return {"ok": True, "workflow_id": WORKFLOW_ID}


@app.post("/transcribe", response_model=JobResponse, responses={202: {"model": EnqueueResponse}})
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    diarize: bool = Query(False),
    wait: bool = Query(False),
):
    # Save uploaded file
    upload_path = storage.allocate_upload_path(file.filename)
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    inputs = {"file_path": str(upload_path), "diarize": diarize}
    job = storage.create_job("transcribe", inputs=inputs)

    if wait:
        # Run inline for blocking behavior
        process_transcribe(job["id"], str(upload_path), diarize)
        job = storage.load_job(job["id"])  # refresh
        return {"job": job}

    # Run asynchronously
    background_tasks.add_task(process_transcribe, job["id"], str(upload_path), diarize)
    return JSONResponse(status_code=202, content=EnqueueResponse(id=job["id"]).model_dump())


@app.post("/summarize", response_model=JobResponse, responses={202: {"model": EnqueueResponse}})
async def summarize(
    background_tasks: BackgroundTasks,
    body: SummarizeRequest,
    wait: bool = Query(False),
):
    # Prepare inputs for LangGraph
    inputs: Dict[str, Any] = {"summary_type": body.summary_type}
    if body.job_id:
        prev_job = storage.load_job(body.job_id)
        if not prev_job:
            raise HTTPException(status_code=404, detail="Referenced job_id not found")
        inputs["job_id"] = body.job_id
    else:
        inputs["text"] = body.text

    job = storage.create_job("summarize", inputs=inputs)

    if wait:
        process_summarize(job["id"], inputs)
        job = storage.load_job(job["id"])  # refresh
        return {"job": job}

    background_tasks.add_task(process_summarize, job["id"], inputs)
    return JSONResponse(status_code=202, content=EnqueueResponse(id=job["id"]).model_dump())


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = storage.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Return entire job
    return {"job": job}


@app.post("/callbacks/langgraph")
async def langgraph_callback(
    payload: LangGraphCallback,
    x_langgraph_secret: Optional[str] = Header(default=None, alias="X-LangGraph-Secret"),
):
    # Validate secret header
    if CALLBACK_SECRET:
        if not x_langgraph_secret or x_langgraph_secret != CALLBACK_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        logger.warning("LANGGRAPH_CALLBACK_SECRET not set; accepting callback without verification")

    job = storage.load_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if payload.status == "completed":
        # Normalize and set result
        result = _extract_job_result({"result": payload.result or {}})
        storage.set_result(payload.job_id, result)
    elif payload.status == "failed":
        storage.set_error(payload.job_id, payload.error or {"message": "Unknown error"})
    else:
        raise HTTPException(status_code=400, detail="Unsupported status")

    return {"ok": True}
