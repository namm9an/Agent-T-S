# LangGraph Workflow Deployment

This document explains how to publish the workflow and wire it to the FastAPI service.

## Files
- Workflow config: `workflows/transcribe_summarize_v1.yaml`

## Prerequisites
- A running LangGraph service (CLI or HTTP API available)
- Qwen API token (for your E2E endpoint) provisioned to the LangGraph runtime as `QWEN_API_KEY`
- FastAPI already running (see README.md) with a public callback URL, or reachable on localhost for development

## Environment Variables (LangGraph runtime)
Required:
- `LANGGRAPH_CALLBACK_SECRET` — Shared secret; must match FastAPI’s `LANGGRAPH_CALLBACK_SECRET`
- `QWEN_API_KEY` — Bearer token for your E2E Qwen endpoint

Optional:
- `JOBS_API_BASE_URL` — Base URL for FastAPI if you want the workflow to fetch transcripts when summarizing with `inputs.job_id` (e.g., `http://localhost:8000`)

## Publish the Workflow

### Using a hypothetical LangGraph CLI
- Authenticate:
  - `langgraph login`
- Publish the workflow file:
  - `langgraph workflows publish workflows/transcribe_summarize_v1.yaml`
- Capture outputs:
  - Workflow ID (e.g., `transcribe_summarize_v1` or a UUID)
  - Trigger URL (HTTP endpoint to start runs)

### Using a hypothetical LangGraph HTTP API
- POST the YAML or an equivalent JSON to your service’s create workflow endpoint. Example payload (pseudo):
  - `POST https://<langgraph-host>/workflows`
  - Body: contents of `workflows/transcribe_summarize_v1.yaml` or JSON equivalent
- The response should contain:
  - `workflow_id` and `trigger_url`

## Wire FastAPI
Set the following in your FastAPI process environment:
- `LANGGRAPH_ENDPOINT=<Trigger URL returned by LangGraph>`
- `LANGGRAPH_WORKFLOW_ID=transcribe_summarize_v1` (or the ID returned if different)
- `LANGGRAPH_CALLBACK_URL=http://<your-fastapi-host>/callbacks/langgraph`
- `LANGGRAPH_CALLBACK_SECRET=<same value used by LangGraph runtime>`
- `MODEL_API_KEY=<Bearer token if your LangGraph trigger endpoint requires auth>`
- `QWEN_ENDPOINT=https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/`
- `WHISPER_ENDPOINT=https://placeholder-whisper-endpoint`

Optional (FastAPI not required to know these, but can forward to LangGraph):
- `QWEN_ENDPOINT` is forwarded in the trigger payload as `qwen_endpoint`
- `WHISPER_ENDPOINT` is forwarded in the trigger payload as `whisper_endpoint`

## Trigger Shape (from FastAPI)
FastAPI will send a payload like:
```json
{
  "workflow_id": "transcribe_summarize_v1",
  "action": "transcribe" | "summarize",
  "job_id": "<uuid-from-fastapi>",
  "inputs": {
    "file_path": "/abs/path/to/file.mp3",   // for transcribe
    "diarize": false,                         // optional
    "summary_type": "executive",             // optional for summarize
    "text": "...",                           // optional (if not using job_id)
    "job_id": "<prev_transcribe_job_id>"     // optional for summarize-from-job
  },
  "callback_url": "http://localhost:8000/callbacks/langgraph",  // optional
  "qwen_endpoint": "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/",
  "whisper_endpoint": "https://placeholder-whisper-endpoint"
}
```

## Callback Shape (from LangGraph to FastAPI)
POST to FastAPI `POST /callbacks/langgraph` with header `X-LangGraph-Secret: <LANGGRAPH_CALLBACK_SECRET>` and body:
```json
{
  "job_id": "<uuid>",
  "status": "completed" | "failed",
  "result": {
    "transcript": "... or null",
    "summary": { "type": "executive", "content": "..." } // structure from summarize_qwen
  },
  "error": null
}
```

## Smoke Tests after Wiring
- Start FastAPI:
  - `uvicorn app.main:app --reload`
- Transcribe (wait=true):
  - `curl -F "file=@/path/to/test.mp3" -F "diarize=false" "http://localhost:8000/transcribe?wait=true"`
- Summarize from text (wait=true):
  - `curl -X POST "http://localhost:8000/summarize?wait=true" -H "Content-Type: application/json" -d '{"text":"Long text","summary_type":"executive"}'`
- Non-blocking + callback:
  - Transcribe without wait, then verify `/jobs/{id}` updates after the callback
- Swagger:
  - `http://localhost:8000/docs`

