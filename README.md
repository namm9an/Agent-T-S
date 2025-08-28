# Transcription + Summarization API (LangGraph v1)

Thin FastAPI layer that delegates the entire workflow to LangGraph:
Audio → Transcription (Whisper hosted) → optional Diarization → Summarization (Qwen-14B 2.5 hosted) → Job storage.

FastAPI never calls models directly. It triggers a LangGraph workflow and tracks job state in files.

---

Requirements
- Python 3.10+
- `pip install -r requirements.txt`

Environment variables
- LANGGRAPH_ENDPOINT: https://<langgraph-service>/trigger (required)
- LANGGRAPH_RESULT_ENDPOINT: https://<langgraph-service>/result (optional; for polling)
- LANGGRAPH_WORKFLOW_ID: transcribe_summarize_v1 (default)
- LANGGRAPH_CALLBACK_URL: https://<this-api>/callbacks/langgraph (optional; if LangGraph pushes results)
- LANGGRAPH_CALLBACK_SECRET: shared secret for callback header X-LangGraph-Secret
- MODEL_API_KEY: Bearer token if LangGraph requires it
- WHISPER_ENDPOINT: https://<your-whisper-service> (optional; forwarded to LangGraph payload)
- QWEN_ENDPOINT: https://infer.e2enetworks.net/project/.../v1/ (optional; forwarded to LangGraph payload)
- JOBS_DIR: ./jobs (default)
- UPLOADS_DIR: ./uploads (default)

You can start with .env.example and export/update variables in your shell.

---

Run
- uvicorn app.main:app --reload

Health check
- GET /health → { ok: true, workflow_id: "..." }

---

Endpoints
1) POST /transcribe
- Form fields: file (required), diarize (bool, default=false), wait (query, default=false)
- Behavior: saves file to ./uploads, creates a job, triggers LangGraph action="transcribe" with { file_path, diarize }.
- If wait=true: runs inline and returns final job if available (or current state if async).
- If wait=false: returns 202 with { id, status } and updates job later via callback or polling.

2) POST /summarize
- JSON body: either { job_id: "..." } or { text: "..." }; also accepts summary_type (default="executive")
- Behavior: creates a job and triggers LangGraph action="summarize".
- wait semantics similar to /transcribe.

3) GET /jobs/{id}
- Returns full job JSON persisted to ./jobs/{id}.json.

4) POST /callbacks/langgraph
- Header: X-LangGraph-Secret must match LANGGRAPH_CALLBACK_SECRET (dev mode accepts if unset and logs a warning).
- Body: { job_id, status: "completed"|"failed", result, error }
- Updates job status and result accordingly.

---

Testing
1) Preflight
- export LANGGRAPH_ENDPOINT=https://your-langgraph/trigger
- export LANGGRAPH_RESULT_ENDPOINT=https://your-langgraph/result   # if using polling
- export LANGGRAPH_WORKFLOW_ID=transcribe_summarize_v1
- export LANGGRAPH_CALLBACK_URL=http://localhost:8000/callbacks/langgraph
- export LANGGRAPH_CALLBACK_SECRET={{LANGGRAPH_CALLBACK_SECRET}}
- export MODEL_API_KEY={{MODEL_API_KEY}}
- export QWEN_ENDPOINT=https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/
- export WHISPER_ENDPOINT=https://<your-whisper-service>
- uvicorn app.main:app --reload

2) Synchronous (if LangGraph returns final result or result endpoint is available)
- curl -F "file=@/path/to/test.mp3" -F "diarize=false" "http://localhost:8000/transcribe?wait=true"

3) Asynchronous with callback
- Start a job:
  curl -F "file=@/path/to/test.mp3" -F "diarize=true" "http://localhost:8000/transcribe"
- Get job:
  curl "http://localhost:8000/jobs/<job_id>"
- Simulate callback (if not yet integrated with LangGraph):
  curl -X POST "http://localhost:8000/callbacks/langgraph" \
    -H "Content-Type: application/json" \
    -H "X-LangGraph-Secret: {{LANGGRAPH_CALLBACK_SECRET}}" \
    -d '{
      "job_id": "<job_id>",
      "status": "completed",
      "result": {
        "transcript": "Hello world transcript",
        "summary": { "type": "executive", "points": ["..."] }
      }
    }'

4) Summarization tests
- From a transcription job:
  curl -X POST "http://localhost:8000/summarize?wait=true" -H "Content-Type: application/json" -d '{"job_id":"<job_id>", "summary_type":"executive"}'
- From raw text:
  curl -X POST "http://localhost:8000/summarize?wait=true" -H "Content-Type: application/json" -d '{"text":"Long text here...", "summary_type":"executive"}'

5) Callback security
- Wrong secret must get 401 Unauthorized.

---

Important: file accessibility
- The API sends file_path to LangGraph. If LangGraph runs remotely, it cannot access local paths. Options:
  1) Use presigned URLs (S3/GCS/MinIO) and pass file_url to LangGraph.
  2) Send the bytes to LangGraph's trigger endpoint (multipart or base64) if supported.
- Tell us which you prefer and we can update the integration accordingly.

---

Job schema (stored in ./jobs/{id}.json)
{
  "id": "<uuid>",
  "type": "transcribe|summarize",
  "status": "queued|running|completed|failed",
  "inputs": { ... },
  "result": { "transcript": "...", "summary": { ... } },
  "error": null,
  "timestamps": { "created_at": "...", "updated_at": "..." }
}

---

Local LangGraph quickstart (CLI)

Terminal A (LangGraph server)
1) Install LangGraph
```
$ pip install langgraph
```
2) Verify installation
```
$ langgraph --help
```
3) Ensure workflow is in place
```
$ mkdir -p workflows
$ mv /path/to/transcribe_summarize_v1.yaml workflows/   # if not already present
```
4) Export env required by the workflow (used inside LangGraph process)
```
# Required by summarize_qwen node
$ export QWEN_API_KEY={{QWEN_API_KEY}}
# Used by maybe_callback_* nodes to sign callbacks
$ export LANGGRAPH_CALLBACK_SECRET=my-secret
```
5) Start LangGraph locally
```
$ langgraph serve --workflows-dir=workflows --port 8080
```

Terminal B (FastAPI)
6) Set FastAPI environment
```
$ export LANGGRAPH_ENDPOINT=http://localhost:8080/workflows/transcribe_summarize_v1/trigger
$ export LANGGRAPH_WORKFLOW_ID=transcribe_summarize_v1
$ export LANGGRAPH_CALLBACK_URL=http://localhost:8000/callbacks/langgraph
$ export LANGGRAPH_CALLBACK_SECRET=my-secret
$ export QWEN_ENDPOINT=https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/
$ export WHISPER_ENDPOINT=https://<your-whisper-endpoint>
# Optional: increase wait timeout for synchronous tests
$ export WAIT_TIMEOUT_SECONDS=180
$ export WAIT_POLL_INTERVAL=1.0
```
7) Start FastAPI
```
$ uvicorn app.main:app --reload --port 8000
```
8) Test Swagger
```
$ open http://localhost:8000/docs
```

Callback smoke test
```
$ curl -X POST "http://localhost:8000/callbacks/langgraph" \
  -H "X-LangGraph-Secret: my-secret" \
  -H "Content-Type: application/json" \
  -d '{"job_id":"<job_id>","status":"completed","result":{"transcript":"Test"}}'
```

Notes
- For /summarize to work, Terminal A must have QWEN_API_KEY exported before starting `langgraph serve`.
- /transcribe returns a placeholder transcript in this v1 workflow; set WHISPER_ENDPOINT for future integration.
- If your LangGraph server exposes a result endpoint, set LANGGRAPH_RESULT_ENDPOINT in FastAPI; otherwise callbacks or immediate responses are used.

---

Qwen endpoint quick test (OpenAI-compatible)
- Set env (do NOT paste keys inline):
  - export E2E_QWEN_BASE_URL=https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/
  - export E2E_QWEN_TOKEN={{E2E_QWEN_TOKEN}}
- Run:
  - python3 scripts/test_qwen.py

Troubleshooting
- status=running for a long time: ensure callback URL and secret are set, or that LANGGRAPH_RESULT_ENDPOINT is configured if using polling.
- status=failed: see the error field; check LangGraph service logs and credentials.
- 401 on callback: verify X-LangGraph-Secret matches LANGGRAPH_CALLBACK_SECRET.

