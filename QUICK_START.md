# Quick Start Guide

## Prerequisites
- Python 3.10+
- E2E Networks API token (already provided in .env)

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Check Configuration
The `.env` file already contains:
- ✅ WHISPER_ENDPOINT: https://infer.e2enetworks.net/project/p-6530/endpoint/is-6351/v1/
- ✅ WHISPER_API_KEY: Your E2E token (already set)
- ✅ QWEN_ENDPOINT: https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/
- ⚠️ QWEN_API_KEY: Add if different from WHISPER_API_KEY

## Step 3: Start Services

### Terminal 1: Start LangGraph
```bash
source .env
python langgraph_server.py
```

### Terminal 2: Start FastAPI
```bash
source .env
uvicorn app.main:app --reload --port 8000
```

## Step 4: Test
Open browser: http://localhost:8000/docs

## That's It!
- Upload audio → Get transcript (from your Whisper)
- Request summary → Get summary (from E2E Qwen)

## NO LOCAL MODELS
Everything runs on remote servers. This just orchestrates the API calls.
