Title

LangGraph-Orchestrated Transcription + Summarization Service — FastAPI API Layer

Purpose

Provide a lightweight FastAPI interface for transcription and summarization where LangGraph runs the full workflow.
FastAPI does not perform transcription or summarization itself; it only:

Accepts audio or text,

Triggers the LangGraph workflow,

Tracks job status,

Exposes results via REST.

Core Workflow

Client uploads audio to /transcribe.

FastAPI saves file, creates a job entry, and triggers LangGraph.

LangGraph orchestrates:

Transcription via hosted Whisper,

Optional diarization (if enabled in workflow),

Summarization via hosted Qwen-14B 2.5.

LangGraph either:

Pushes the result back via callback, or

Makes it available for polling.

FastAPI updates the job and serves it via /jobs/{id}.

API Endpoints
POST /transcribe

Input: multipart/form-data with:

file: audio file,

diarize: boolean (optional),

wait: boolean (optional, wait up to timeout for result).

Output:

Non-blocking: 202 { "id": "<job_id>", "status": "queued" }

Blocking (?wait=true): full job JSON when workflow completes.

POST /summarize

Input: JSON with either:

{"job_id":"<id>"} — uses transcript from transcription job, or

{"text":"..."}

Query summary_type (default: executive),

wait boolean (optional).

Output: same as /transcribe.

GET /jobs/{id}

Output: Full job state:

{
  "id": "<uuid>",
  "type": "transcribe|summarize",
  "status": "queued|running|completed|failed",
  "inputs": {...},
  "result": {"transcript": "...", "summary": {...}},
  "error": null,
  "timestamps": {"created_at": "...", "updated_at": "..."}
}

Environment Variables
JOBS_DIR=./jobs
LANGGRAPH_ENDPOINT=https://<your-langgraph-service>/trigger
LANGGRAPH_RESULT_ENDPOINT=https://<your-langgraph-service>/result   # optional, if polling is used
LANGGRAPH_WORKFLOW_ID=transcribe_summarize_v1
MODEL_API_KEY=<if required by LangGraph>

Behavior

LangGraph is the orchestrator — it runs Whisper and Qwen agents.

FastAPI is thin — it does not host or run Whisper/Qwen models.

Job state persisted in files — ./jobs/{id}.json.

Async & Wait

Background tasks trigger the workflow and return immediately.

wait=true allows blocking for up to a defined timeout for Swagger/local testing.

Error Handling

If LangGraph call fails, job is marked failed with error details.

If LangGraph workflow fails, same outcome.

Security

Development: open access for Swagger testing.

Production: add API key middleware, size limits, and HTTPS.

Diarization

Controlled by diarize parameter on /transcribe.

Implementation depends on LangGraph workflow. FastAPI does not modify audio.

Workflow Responsibilities

FastAPI: receive job, persist metadata, call LangGraph, track state.

LangGraph: run Whisper transcription, diarize (optional), run Qwen summarization, return result.

Testing

Start FastAPI:

uvicorn app.main:app --reload


Call /transcribe?wait=true with sample audio.

Check /jobs/{id} for state.

Call /summarize?wait=true with returned job ID if workflow does not auto-chain summarization.

Milestones

Confirm LangGraph workflow connectivity and job lifecycle.

Validate that transcription and summarization outputs from LangGraph map to expected fields.

Implement callback handler if LangGraph supports push updates; otherwise rely on polling.

Out of Scope (v1)

Local Whisper/Qwen execution.

Docker packaging (dev only).

Redis/Celery queues.