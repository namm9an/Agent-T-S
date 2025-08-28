import os
import json
import base64
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# E2E imports for Whisper
try:
    from e2enetworks.cloud import tir
    tir.init()
    TIR_CLIENT = tir.ModelAPIClient()
except Exception:
    TIR_CLIENT = None

app = FastAPI(title="Mock LangGraph (dev only)")

QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
CALLBACK_SECRET = os.getenv("LANGGRAPH_CALLBACK_SECRET")


@app.post("/workflows/transcribe_summarize_v1/trigger")
async def trigger(req: Request):
    payload: Dict[str, Any] = await req.json()
    action = payload.get("action")
    job_id = payload.get("job_id")
    inputs = payload.get("inputs", {})
    callback_url = payload.get("callback_url")

    if action == "transcribe":
        file_path = inputs.get("file_path")
        diarize = bool(inputs.get("diarize", False))
        transcript = None
        error = None
        
        # Try real transcription with E2E
        if TIR_CLIENT and file_path and os.path.exists(file_path):
            try:
                # Read the audio file
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
                
                # E2E expects base64 for binary input
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                data = {
                    "input": audio_b64,  # base64 encoded audio
                    "language": "English",
                    "task": "transcribe",
                    "max_new_tokens": 400,
                    "return_timestamps": "none"
                }
                
                output = TIR_CLIENT.infer(model_name="whisper-large-v3", data=data)
                transcript = output.get("text") or output.get("transcript") or str(output)
            except Exception as e:
                error = f"Whisper error: {str(e)}"
                transcript = f"[ERROR] Failed to transcribe {file_path}"
        else:
            # Fallback placeholder
            transcript = f"[PLACEHOLDER] Transcribed {file_path}; diarize={diarize}"
        
        result = {
            "job_id": job_id,
            "status": "completed" if not error else "failed",
            "result": {"transcript": transcript, "summary": None},
            "error": error,
        }
        # Optionally push callback (best-effort)
        if callback_url:
            headers = {"Content-Type": "application/json"}
            if CALLBACK_SECRET:
                headers["X-LangGraph-Secret"] = CALLBACK_SECRET
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(callback_url, json=result, headers=headers)
            except Exception:
                pass
        return JSONResponse(result)

    if action == "summarize":
        summary_type = (inputs.get("summary_type") or "executive")
        text = inputs.get("text") or ""
        content: Any = None
        error: Any = None

        if QWEN_ENDPOINT and QWEN_API_KEY:
            try:
                payload_qwen = {
                    "model": "Qwen/Qwen2.5-14B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that produces concise summaries."},
                        {
                            "role": "user",
                            "content": (
                                f"Provide a {summary_type} summary for the following text.\n"
                                "Return concise bullet points in JSON under keys: {\"type\": \"<type>\", \"bullets\": [\"...\"]}.\n"
                                f"Text:\n{text}"
                            ),
                        },
                    ],
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        QWEN_ENDPOINT.rstrip("/") + "/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {QWEN_API_KEY}",
                        },
                        json=payload_qwen,
                    )
                    r.raise_for_status()
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
            except Exception as e:
                error = str(e)

        if content is None:
            # Fallback stub content
            content = json.dumps({"type": summary_type, "bullets": ["(stub) summary"]})

        result = {
            "job_id": job_id,
            "status": "completed",
            "result": {
                "transcript": None,
                "summary": {"type": summary_type, "content": content},
            },
            "error": error,
        }
        # Optionally push callback (best-effort)
        if callback_url:
            headers = {"Content-Type": "application/json"}
            if CALLBACK_SECRET:
                headers["X-LangGraph-Secret"] = CALLBACK_SECRET
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(callback_url, json=result, headers=headers)
            except Exception:
                pass
        return JSONResponse(result)

    return JSONResponse({"error": "Unsupported action"}, status_code=400)

