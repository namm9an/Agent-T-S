# System Architecture - NO LOCAL MODELS

## ⚠️ IMPORTANT: All Models Run on Remote Servers

**This system does NOT download or run any AI models locally.** It only orchestrates API calls to remote endpoints.

## What Runs Where:

### 🖥️ On Your Local Machine (or E2E Node):
1. **FastAPI** (Port 8000) - API layer only
2. **LangGraph** (Port 8123) - Workflow orchestration only
3. **File Storage** - Temporary audio files and job JSONs only

### ☁️ On Remote Servers:
1. **Whisper Model** - Your vLLM server (you provide the endpoint)
2. **Qwen 14B Model** - E2E Networks server (already provided)

## How It Works:

```
[Client Upload Audio] 
    ↓
[FastAPI saves file locally - just the audio file, no models]
    ↓
[LangGraph orchestrates the workflow]
    ↓
[LangGraph calls REMOTE Whisper API] ← Your vLLM endpoint
    ↓
[LangGraph calls REMOTE Qwen API] ← E2E Networks endpoint  
    ↓
[Results returned to client]
```

## What You Need to Provide:

1. **WHISPER_ENDPOINT**: URL of your remote Whisper vLLM server
   - Example: `https://your-whisper-server.com/v1`
   
2. **QWEN_API_KEY**: Your E2E Networks API key (if required)

## Storage Clarification:

The `./jobs` and `./uploads` directories store:
- **jobs/**: JSON files with job status (few KB each)
- **uploads/**: Temporary audio files from users

They do NOT store:
- ❌ AI models
- ❌ Model weights
- ❌ Training data

## Resource Requirements:

### For Running This System:
- CPU: 1-2 cores
- RAM: 1-2 GB
- Disk: 1 GB (for audio files)
- GPU: NOT REQUIRED ❌

### For Running Models (Remote Servers):
- Whisper: Runs on YOUR remote vLLM server
- Qwen: Runs on E2E's GPU servers

## Summary:

This is a **lightweight orchestration system** that:
- ✅ Accepts user uploads
- ✅ Manages job queues
- ✅ Calls remote AI APIs
- ❌ Does NOT run AI models locally
- ❌ Does NOT need GPU
- ❌ Does NOT download model weights
