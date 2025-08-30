#!/usr/bin/env python3
"""
Integrated FastAPI + LangGraph Service
This combines everything into one service with embedded LangGraph workflow
"""
import os
import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import httpx
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6352/v1/")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")

# Storage paths
JOBS_DIR = Path(os.getenv("JOBS_DIR", "./jobs"))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "./uploads"))
JOBS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LangGraph Workflow State
# ============================================================================

class WorkflowState(BaseModel):
    """State that flows through the LangGraph workflow"""
    job_id: str
    action: str  # "transcribe" or "summarize"
    file_path: Optional[str] = None
    text: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    summary_type: str = "executive"
    error: Optional[str] = None
    status: str = "running"
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# LangGraph Agent Nodes
# ============================================================================

async def transcribe_agent(state: WorkflowState) -> Dict[str, Any]:
    """Whisper transcription agent"""
    try:
        if not state.file_path or not os.path.exists(state.file_path):
            state.error = f"File not found: {state.file_path}"
            state.status = "failed"
            return state.dict()
        
        logger.info(f"Transcribing file: {state.file_path}")
        
        # Use E2E Networks SDK for Whisper transcription
        if WHISPER_API_KEY:
            try:
                # Initialize E2E Networks client
                os.environ['E2E_TIR_ACCESS_TOKEN'] = WHISPER_API_KEY
                os.environ['E2E_TIR_PROJECT_ID'] = 'p-6530'
                
                from e2enetworks.cloud import tir
                tir.init()
                client = tir.ModelAPIClient()
                
                # Read audio file and encode as base64
                with open(state.file_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                with open(state.file_path, 'rb') as audio_file:
                    files = {
                        'file': (os.path.basename(state.file_path), audio_file.read(), 'audio/mpeg')
                    }
                    data = {
                        'model': 'whisper-1',
                        'response_format': 'json'
                    }
                    headers = {
                        "Authorization": f"Bearer {WHISPER_API_KEY}"
                    }
                    
                # Prepare data for Whisper model
                data = {
                    "input": audio_base64,  # Base64 encoded audio
                    "language": "English",
                    "task": "transcribe",
                    "max_new_tokens": 4096,  # Increased for longer audio
                    "return_timestamps": "none"
                }
                
                logger.info(f"Calling E2E Whisper model...")
                
                # Call the model in async context
                import asyncio
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(None, client.infer, "whisper-large-v3", data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            state.transcript = result.get("text", "")
                            state.status = "completed"
                            logger.info(f"Transcription successful, {len(state.transcript)} characters")
                        else:
                            # If audio endpoint fails, try chat completion as fallback
                            logger.warning(f"Audio endpoint failed: {response.status_code}, trying chat completion")
                            
                            chat_url = f"{WHISPER_ENDPOINT.rstrip('/')}/chat/completions"
                            # For Whisper configured as chat model
                            payload = {
                                "model": "openai/whisper-large-v3",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": f"Transcribe the audio file: {os.path.basename(state.file_path)}"
                                    }
                                ],
                                "temperature": 0.0,
                                "max_tokens": 4000
                            }
                            
                            response = await client.post(
                                chat_url,
                                json=payload,
                                headers={"Authorization": f"Bearer {WHISPER_API_KEY}", "Content-Type": "application/json"}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                # Note: This won't be actual transcription if Whisper is chat-only
                                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                state.transcript = f"[Whisper API Mode Issue] {content}"
                                state.status = "completed"
                            else:
                                state.transcript = f"[API Error {response.status_code}]"
                                state.error = f"Whisper API error: {response.text}"
                                state.status = "failed"
                    
                    except Exception as e:
                        logger.error(f"Whisper API call failed: {e}")
                        state.error = str(e)
                        state.status = "failed"
        else:
            # Fallback for testing
            state.transcript = f"[Test Mode] Would transcribe: {os.path.basename(state.file_path)}"
            state.status = "completed"
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        state.error = str(e)
        state.status = "failed"
    
    return state.dict()

async def summarize_agent(state: WorkflowState) -> Dict[str, Any]:
    """Qwen summarization agent"""
    try:
        # Get text to summarize
        text_to_summarize = state.text or state.transcript
        
        if not text_to_summarize:
            state.error = "No text available for summarization"
            state.status = "failed"
            return state.dict()
        
        logger.info(f"Summarizing {len(text_to_summarize)} characters")
        
        # Call Qwen API for summarization
        if QWEN_ENDPOINT and QWEN_API_KEY:
            async with httpx.AsyncClient(timeout=30.0) as client:
                chat_url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
                
                system_prompt = "You are a helpful assistant that produces concise summaries."
                user_prompt = f"""Provide a {state.summary_type} summary for the following text.
Return concise bullet points in JSON format with keys: {{"type": "<type>", "bullets": ["..."]}}

Text:
{text_to_summarize}"""
                
                payload = {
                    "model": "Qwen/Qwen2.5-14B-Instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                headers = {
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post(chat_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    # Try to parse as JSON
                    try:
                        summary_json = json.loads(content)
                    except:
                        summary_json = {
                            "type": state.summary_type,
                            "content": content
                        }
                    
                    state.summary = summary_json
                    state.status = "completed"
                    logger.info("Summarization successful")
                else:
                    state.error = f"Qwen API error: {response.status_code}"
                    state.status = "failed"
        else:
            # Fallback for testing
            state.summary = {
                "type": state.summary_type,
                "content": "[Test Mode] Summary would appear here"
            }
            state.status = "completed"
    
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        state.error = str(e)
        state.status = "failed"
    
    return state.dict()

# ============================================================================
# Build LangGraph Workflow
# ============================================================================

def create_workflow() -> StateGraph:
    """Create the LangGraph agent workflow"""
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes (agents)
    workflow.add_node("transcribe", transcribe_agent)
    workflow.add_node("summarize", summarize_agent)
    
    # Define the routing logic
    def route_action(state: WorkflowState) -> str:
        if state.action == "transcribe":
            return "transcribe"
        elif state.action == "summarize":
            return "summarize"
        else:
            return END
    
    # Add a simple routing node
    workflow.add_node("route", lambda x: x)
    workflow.set_entry_point("route")
    
    # Add conditional routing from the route node
    workflow.add_conditional_edges(
        "route",
        route_action,
        {
            "transcribe": "transcribe",
            "summarize": "summarize",
            END: END
        }
    )
    
    # Transcribe can optionally lead to summarize
    workflow.add_edge("transcribe", END)
    workflow.add_edge("summarize", END)
    
    return workflow

# Create the workflow once
checkpointer = MemorySaver()
langgraph_workflow = create_workflow().compile(checkpointer=checkpointer)

# ============================================================================
# Job Storage Functions
# ============================================================================

def create_job(job_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new job"""
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "inputs": inputs,
        "result": {"transcript": None, "summary": None},
        "error": None,
        "timestamps": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    }
    
    # Save job to file
    job_path = JOBS_DIR / f"{job_id}.json"
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    
    return job

def update_job(job_id: str, **updates) -> Optional[Dict[str, Any]]:
    """Update an existing job"""
    job_path = JOBS_DIR / f"{job_id}.json"
    
    if not job_path.exists():
        return None
    
    with open(job_path, "r") as f:
        job = json.load(f)
    
    # Update fields
    for key, value in updates.items():
        if key == "result" and isinstance(value, dict):
            job.setdefault("result", {})
            job["result"].update(value)
        else:
            job[key] = value
    
    job["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    # Save back
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    
    return job

def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Load a job from storage"""
    job_path = JOBS_DIR / f"{job_id}.json"
    
    if not job_path.exists():
        return None
    
    with open(job_path, "r") as f:
        return json.load(f)

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="LangGraph AI Agent - Transcription & Summarization",
    description="Integrated FastAPI + LangGraph service with Whisper and Qwen agents"
)

# ============================================================================
# Request/Response Models
# ============================================================================

class SummarizeRequest(BaseModel):
    job_id: Optional[str] = None
    text: Optional[str] = None
    summary_type: str = "executive"

class JobResponse(BaseModel):
    job: Dict[str, Any]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LangGraph AI Agent",
        "agents": ["whisper-transcription", "qwen-summarization"]
    }

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    wait: bool = Query(False)
):
    """Transcribe audio file using Whisper agent"""
    
    # Save uploaded file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Create job
    inputs = {"file_path": str(file_path)}
    job = create_job("transcribe", inputs)
    
    # Create workflow state
    state = WorkflowState(
        job_id=job["id"],
        action="transcribe",
        file_path=str(file_path)
    )
    
    if wait:
        # Run synchronously
        config = {"configurable": {"thread_id": job["id"]}}
        result = await langgraph_workflow.ainvoke(state.dict(), config=config)
        
        # Update job with results
        update_job(
            job["id"],
            status=result.get("status", "completed"),
            result={"transcript": result.get("transcript"), "summary": None},
            error=result.get("error")
        )
        
        return {"job": load_job(job["id"])}
    else:
        # Run asynchronously
        async def run_async():
            config = {"configurable": {"thread_id": job["id"]}}
            result = await langgraph_workflow.ainvoke(state.dict(), config=config)
            update_job(
                job["id"],
                status=result.get("status", "completed"),
                result={"transcript": result.get("transcript"), "summary": None},
                error=result.get("error")
            )
        
        background_tasks.add_task(run_async)
        return JSONResponse(
            status_code=202,
            content={"id": job["id"], "status": "queued"}
        )

@app.post("/summarize")
async def summarize(
    body: SummarizeRequest,
    background_tasks: BackgroundTasks,
    wait: bool = Query(False)
):
    """Summarize text using Qwen agent"""
    
    # Prepare text
    text = body.text
    if body.job_id:
        prev_job = load_job(body.job_id)
        if not prev_job:
            raise HTTPException(status_code=404, detail="Job not found")
        text = prev_job.get("result", {}).get("transcript")
        if not text:
            raise HTTPException(status_code=400, detail="No transcript found in job")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Create job
    inputs = {"text": text, "summary_type": body.summary_type}
    if body.job_id:
        inputs["source_job_id"] = body.job_id
    job = create_job("summarize", inputs)
    
    # Create workflow state
    state = WorkflowState(
        job_id=job["id"],
        action="summarize",
        text=text,
        summary_type=body.summary_type
    )
    
    if wait:
        # Run synchronously
        config = {"configurable": {"thread_id": job["id"]}}
        result = await langgraph_workflow.ainvoke(state.dict(), config=config)
        
        # Update job with results
        update_job(
            job["id"],
            status=result.get("status", "completed"),
            result={"transcript": None, "summary": result.get("summary")},
            error=result.get("error")
        )
        
        return {"job": load_job(job["id"])}
    else:
        # Run asynchronously
        async def run_async():
            config = {"configurable": {"thread_id": job["id"]}}
            result = await langgraph_workflow.ainvoke(state.dict(), config=config)
            update_job(
                job["id"],
                status=result.get("status", "completed"),
                result={"transcript": None, "summary": result.get("summary")},
                error=result.get("error")
            )
        
        background_tasks.add_task(run_async)
        return JSONResponse(
            status_code=202,
            content={"id": job["id"], "status": "queued"}
        )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results"""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    summary_type: str = Query("executive")
):
    """Process audio file: transcribe and summarize in one go"""
    
    # Save uploaded file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Create job
    inputs = {"file_path": str(file_path), "summary_type": summary_type}
    job = create_job("process", inputs)
    
    # Step 1: Transcribe
    transcribe_state = WorkflowState(
        job_id=job["id"],
        action="transcribe",
        file_path=str(file_path)
    )
    
    config = {"configurable": {"thread_id": job["id"]}}
    transcribe_result = await langgraph_workflow.ainvoke(transcribe_state.dict(), config=config)
    
    if transcribe_result.get("status") == "failed":
        update_job(job["id"], status="failed", error=transcribe_result.get("error"))
        return {"job": load_job(job["id"])}
    
    # Step 2: Summarize
    summarize_state = WorkflowState(
        job_id=job["id"],
        action="summarize",
        text=transcribe_result.get("transcript"),
        summary_type=summary_type
    )
    
    summarize_result = await langgraph_workflow.ainvoke(summarize_state.dict(), config=config)
    
    # Update job with both results
    update_job(
        job["id"],
        status="completed",
        result={
            "transcript": transcribe_result.get("transcript"),
            "summary": summarize_result.get("summary")
        },
        error=summarize_result.get("error")
    )
    
    return {"job": load_job(job["id"])}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Check environment variables
    if not WHISPER_API_KEY:
        logger.warning("WHISPER_API_KEY not set - transcription will use placeholder")
    if not QWEN_API_KEY:
        logger.warning("QWEN_API_KEY not set - summarization will use placeholder")
    
    logger.info("Starting integrated LangGraph AI Agent service...")
    logger.info(f"Whisper endpoint: {WHISPER_ENDPOINT}")
    logger.info(f"Qwen endpoint: {QWEN_ENDPOINT}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
