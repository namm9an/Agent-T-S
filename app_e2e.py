#!/usr/bin/env python3
"""
Integrated FastAPI + LangGraph with E2E Networks Whisper
"""
import os
import json
import uuid
import asyncio
import logging
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")

# Storage paths
JOBS_DIR = Path("./jobs")
UPLOADS_DIR = Path("./uploads")
JOBS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Global job progress tracker
job_progress = {}

# ============================================================================
# Workflow State
# ============================================================================

class WorkflowState(BaseModel):
    job_id: str
    action: str
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
# Audio Utilities
# ============================================================================

def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
    return 0

async def split_audio_fast(file_path: str, chunk_duration: int = 25, overlap: int = 2) -> List[str]:
    """Split audio file into chunks with overlap for better transcription continuity"""
    chunks = []
    duration = get_audio_duration(file_path)
    
    if duration <= 30:  # E2E limit is 30s
        return [file_path]  # No need to split
    
    try:
        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
        base_name = Path(file_path).stem
        ext = Path(file_path).suffix
        
        # Calculate chunks with overlap
        effective_chunk_duration = chunk_duration - overlap
        num_chunks = int((duration - overlap) / effective_chunk_duration) + 1
        
        logger.info(f"Splitting {file_path} (duration: {duration:.1f}s) into {num_chunks} chunks with {overlap}s overlap")
        
        # Create all ffmpeg commands first
        commands = []
        for i in range(num_chunks):
            start_time = max(0, i * effective_chunk_duration)
            chunk_path = temp_dir / f"{base_name}_chunk_{i:03d}{ext}"
            
            # Optimized ffmpeg command for speed
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-ss', str(start_time),  # Seek before input for speed
                '-i', file_path,
                '-t', str(chunk_duration),
                '-c:a', 'libmp3lame',  # Fast MP3 encoding
                '-b:a', '128k',  # Lower bitrate for faster processing
                '-ar', '16000',  # Downsample to 16kHz (Whisper works fine with this)
                '-ac', '1',  # Convert to mono
                str(chunk_path)
            ]
            commands.append((i, chunk_path, cmd))
        
        # Process chunks in parallel using asyncio
        async def process_chunk(idx, chunk_path, cmd):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                
                if proc.returncode == 0 and chunk_path.exists():
                    logger.info(f"Created chunk {idx+1}/{num_chunks}")
                    return str(chunk_path)
                else:
                    logger.error(f"Failed chunk {idx}: {stderr.decode()}")
                    return None
            except asyncio.TimeoutError:
                logger.error(f"Timeout on chunk {idx}")
                return None
        
        # Run all chunks in parallel
        tasks = [process_chunk(idx, path, cmd) for idx, path, cmd in commands]
        results = await asyncio.gather(*tasks)
        
        # Filter out failed chunks
        chunks = [chunk for chunk in results if chunk is not None]
        
        if not chunks:
            logger.warning("No chunks created, falling back to original file")
            return [file_path]
            
        logger.info(f"Successfully created {len(chunks)} chunks in parallel")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to split audio: {e}")
        return [file_path]  # Return original file on error

# ============================================================================
# Agents
# ============================================================================

async def transcribe_agent(state: WorkflowState) -> Dict[str, Any]:
    """OpenAI-style Whisper transcription with parallel chunk processing"""
    try:
        if not state.file_path or not os.path.exists(state.file_path):
            state.error = f"File not found: {state.file_path}"
            state.status = "failed"
            return state.model_dump()
        
        logger.info(f"Starting transcription: {state.file_path}")
        start_time = asyncio.get_event_loop().time()
        
        if WHISPER_API_KEY:
            import openai
            
            # Configure OpenAI client for E2E Whisper
            client = openai.OpenAI(
                api_key=WHISPER_API_KEY,
                base_url="https://infer.e2enetworks.net/project/p-6530/endpoint/is-6356/v1/"
            )
            
            # Split audio if needed
            chunks = await split_audio_fast(state.file_path)
            temp_dir = Path(chunks[0]).parent if len(chunks) > 1 else None
            
            # Process chunks in parallel batches to avoid overwhelming the API
            batch_size = 3  # Process 3 chunks at a time
            transcripts = []
            
            async def transcribe_chunk(chunk_path: str, chunk_idx: int) -> Optional[str]:
                """Transcribe a single chunk asynchronously"""
                try:
                    logger.info(f"Transcribing chunk {chunk_idx+1}/{len(chunks)}")
                    with open(chunk_path, 'rb') as audio_file:
                        loop = asyncio.get_event_loop()
                        transcription = await loop.run_in_executor(
                            None,
                            lambda: client.audio.transcriptions.create(
                                model="openai/whisper-large-v3",
                                file=audio_file,
                                response_format="text",
                                language="en"  # Specify language to speed up
                            )
                        )
                        
                        if hasattr(transcription, 'text'):
                            return transcription.text
                        else:
                            return str(transcription)
                            
                except Exception as e:
                    logger.error(f"Failed chunk {chunk_idx+1}: {str(e)[:100]}")
                    return None
            
            # Track progress
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            job_progress[state.job_id] = {
                "status": "transcribing",
                "total_chunks": len(chunks),
                "completed_chunks": 0,
                "total_batches": total_batches,
                "completed_batches": 0
            }
            
            # Process in batches
            for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                # Create tasks for this batch
                tasks = [
                    transcribe_chunk(chunk, batch_start + i)
                    for i, chunk in enumerate(batch_chunks)
                ]
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks)
                
                # Add successful transcriptions
                transcripts.extend([t for t in batch_results if t is not None])
                
                # Update progress
                completed_chunks = batch_end
                job_progress[state.job_id].update({
                    "completed_chunks": completed_chunks,
                    "completed_batches": batch_idx + 1,
                    "percentage": int((completed_chunks / len(chunks)) * 100)
                })
                
                logger.info(f"Completed batch {batch_idx + 1}/{total_batches} ({completed_chunks}/{len(chunks)} chunks)")
            
            # Clean up temp files
            if temp_dir:
                try:
                    for chunk in chunks:
                        Path(chunk).unlink(missing_ok=True)
                    temp_dir.rmdir()
                except:
                    pass
            
            # Process overlapping chunks to remove duplicates
            if len(chunks) > 1 and transcripts:
                # Simple approach: just join with space
                # In production, you'd want smarter overlap handling
                state.transcript = " ".join(transcripts)
            elif transcripts:
                state.transcript = transcripts[0]
            else:
                state.error = "Failed to transcribe any audio chunks"
                state.status = "failed"
                return state.model_dump()
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Transcription complete: {len(state.transcript)} chars in {elapsed:.1f}s")
            state.status = "completed"
            
            # Clean up progress tracking
            if state.job_id in job_progress:
                del job_progress[state.job_id]
                    
        else:
            state.transcript = f"[No API Key] Test transcript for {os.path.basename(state.file_path)}"
            state.status = "completed"
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        state.error = str(e)
        state.status = "failed"
    
    return state.model_dump()

async def summarize_agent(state: WorkflowState) -> Dict[str, Any]:
    """Qwen summarization"""
    try:
        text = state.text or state.transcript
        if not text:
            state.error = "No text to summarize"
            state.status = "failed"
            return state.model_dump()
        
        logger.info(f"Summarizing {len(text)} characters")
        
        if QWEN_ENDPOINT and QWEN_API_KEY:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions",
                    json={
                        "model": "Qwen/Qwen2.5-14B-Instruct",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that produces concise summaries."},
                            {"role": "user", "content": f"Summarize this text in bullet points:\n\n{text}"}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    state.summary = {"type": state.summary_type, "content": content}
                    state.status = "completed"
                else:
                    state.error = f"Qwen error: {response.status_code}"
                    state.status = "failed"
        else:
            state.summary = {"type": state.summary_type, "content": "[No API Key] Test summary"}
            state.status = "completed"
            
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        state.error = str(e)
        state.status = "failed"
    
    return state.model_dump()

# ============================================================================
# Workflow
# ============================================================================

def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("transcribe", transcribe_agent)
    workflow.add_node("summarize", summarize_agent)
    
    def route(state: WorkflowState) -> str:
        if state.action == "transcribe":
            return "transcribe"
        elif state.action == "summarize":
            return "summarize"
        return END
    
    workflow.add_node("route", lambda x: x)
    workflow.set_entry_point("route")
    
    workflow.add_conditional_edges(
        "route", route,
        {"transcribe": "transcribe", "summarize": "summarize", END: END}
    )
    
    workflow.add_edge("transcribe", END)
    workflow.add_edge("summarize", END)
    
    return workflow

langgraph_workflow = create_workflow().compile(checkpointer=MemorySaver())

# ============================================================================
# Job Management
# ============================================================================

def create_job(job_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job, f, indent=2)
    return job

def update_job(job_id: str, **updates) -> None:
    job_path = JOBS_DIR / f"{job_id}.json"
    if job_path.exists():
        with open(job_path, "r") as f:
            job = json.load(f)
        for key, value in updates.items():
            if key == "result" and isinstance(value, dict):
                job["result"].update(value)
            else:
                job[key] = value
        job["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(job_path, "w") as f:
            json.dump(job, f, indent=2)

def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    job_path = JOBS_DIR / f"{job_id}.json"
    if job_path.exists():
        with open(job_path, "r") as f:
            return json.load(f)
    return None

# ============================================================================
# FastAPI
# ============================================================================

app = FastAPI(title="LangGraph AI Agent with E2E Whisper")

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": ["e2e-whisper", "qwen"]}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), wait: bool = Query(False)):
    # Save file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Create job
    job = create_job("transcribe", {"file_path": str(file_path)})
    
    # Run workflow
    state = WorkflowState(
        job_id=job["id"],
        action="transcribe",
        file_path=str(file_path)
    )
    
    result = await langgraph_workflow.ainvoke(
        state.model_dump(),
        config={"configurable": {"thread_id": job["id"]}}
    )
    
    # Update job
    update_job(
        job["id"],
        status=result.get("status"),
        result={"transcript": result.get("transcript")},
        error=result.get("error")
    )
    
    return {"job": load_job(job["id"])}

@app.post("/transcribe-summarize")
async def transcribe_and_summarize(file: UploadFile = File(...), summary_type: str = Query("executive")):
    """Transcribe audio and then summarize the transcript"""
    # Save file
    file_path = UPLOADS_DIR / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Create job
    job = create_job("transcribe-summarize", {
        "file_path": str(file_path),
        "summary_type": summary_type
    })
    
    # Step 1: Transcribe
    state = WorkflowState(
        job_id=job["id"],
        action="transcribe",
        file_path=str(file_path)
    )
    
    transcribe_result = await langgraph_workflow.ainvoke(
        state.model_dump(),
        config={"configurable": {"thread_id": f"{job['id']}_transcribe"}}
    )
    
    if transcribe_result.get("status") == "failed":
        update_job(
            job["id"],
            status="failed",
            error=transcribe_result.get("error", "Transcription failed")
        )
        return {"job": load_job(job["id"])}
    
    # Step 2: Summarize the transcript
    state = WorkflowState(
        job_id=job["id"],
        action="summarize",
        text=transcribe_result.get("transcript"),
        summary_type=summary_type
    )
    
    summarize_result = await langgraph_workflow.ainvoke(
        state.model_dump(),
        config={"configurable": {"thread_id": f"{job['id']}_summarize"}}
    )
    
    # Update job with both results
    update_job(
        job["id"],
        status=summarize_result.get("status"),
        result={
            "transcript": transcribe_result.get("transcript"),
            "summary": summarize_result.get("summary")
        },
        error=summarize_result.get("error")
    )
    
    return {"job": load_job(job["id"])}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Add progress info if available
    if job_id in job_progress:
        job["progress"] = job_progress[job_id]
    
    return {"job": job}

@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get real-time progress for a transcription job"""
    if job_id in job_progress:
        return {"progress": job_progress[job_id]}
    
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    return {
        "progress": {
            "status": job.get("status", "unknown"),
            "completed": job.get("status") in ["completed", "failed"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting E2E Whisper + Qwen service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
