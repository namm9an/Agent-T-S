#!/usr/bin/env python3
"""
LangGraph Application for Transcription + Summarization Workflow
This runs the workflow defined in workflows/transcribe_summarize_v1.yaml
"""

import asyncio
import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import httpx
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "https://infer.e2enetworks.net/project/p-6530/endpoint/is-6351/v1/")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")
LANGGRAPH_CALLBACK_SECRET = os.getenv("LANGGRAPH_CALLBACK_SECRET", "")
JOBS_API_BASE_URL = os.getenv("JOBS_API_BASE_URL", "http://localhost:8000")

# State definition
class WorkflowState(BaseModel):
    action: str
    job_id: str
    inputs: Dict[str, Any]
    callback_url: Optional[str] = None
    qwen_endpoint: Optional[str] = None
    whisper_endpoint: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    text: Optional[str] = None
    summary_type: Optional[str] = "executive"

def route_by_action(state: WorkflowState) -> str:
    """Route based on the action type"""
    if state.action == "transcribe":
        return "transcribe_placeholder"
    elif state.action == "summarize":
        return "summarize_prepare_text"
    else:
        return "error"

async def transcribe_placeholder(state: WorkflowState) -> WorkflowState:
    """Real transcription using Whisper audio API"""
    try:
        file_path = state.inputs.get("file_path")
        diarize = state.inputs.get("diarize", False)
        
        # Use the Whisper endpoint if available
        endpoint = state.whisper_endpoint or WHISPER_ENDPOINT
        if endpoint and WHISPER_API_KEY and os.path.exists(file_path):
            # Whisper expects actual audio files via multipart form
            headers = {
                "Authorization": f"Bearer {WHISPER_API_KEY}"
            }
            
            # Read the audio file
            with open(file_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(file_path), audio_file, 'audio/wav'),
                }
                data = {
                    'model': 'openai/whisper-large-v3',
                    'response_format': 'json',
                    'language': 'en'  # Auto-detect if not specified
                }
                
                if diarize:
                    data['diarize'] = 'true'
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{endpoint.rstrip('/')}/audio/transcriptions",
                        files=files,
                        data=data,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        transcript = result.get("text", "")
                        logger.info(f"Transcription completed for {file_path}")
                    else:
                        logger.error(f"Whisper API error: {response.status_code} - {response.text}")
                        transcript = f"[ERROR] Whisper API returned {response.status_code}"
        else:
            # Fallback to placeholder if file doesn't exist or no endpoint
            if not os.path.exists(file_path):
                transcript = f"[ERROR] File not found: {file_path}"
            else:
                transcript = f"[PLACEHOLDER] Would transcribe {file_path}; diarize={diarize}"
        
        state.result = {
            "transcript": transcript,
            "summary": None
        }
        return state
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        state.error = str(e)
        return state

async def summarize_prepare_text(state: WorkflowState) -> WorkflowState:
    """Prepare text for summarization"""
    try:
        text = state.inputs.get("text")
        from_job_id = state.inputs.get("job_id")
        summary_type = state.inputs.get("summary_type", "executive")
        
        state.summary_type = summary_type
        
        if text:
            state.text = text
            return state
        
        # Try to fetch transcript from previous job
        if from_job_id and JOBS_API_BASE_URL:
            async with httpx.AsyncClient() as client:
                url = f"{JOBS_API_BASE_URL.rstrip('/')}/jobs/{from_job_id}"
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    job_data = data.get("job", {})
                    transcript = job_data.get("result", {}).get("transcript")
                    if transcript:
                        state.text = transcript
                        return state
        
        raise RuntimeError("No text available for summarization")
        
    except Exception as e:
        state.error = str(e)
        return state

async def summarize_qwen(state: WorkflowState) -> WorkflowState:
    """Call Qwen API for summarization"""
    try:
        if not state.text:
            raise RuntimeError("No text to summarize")
        
        endpoint = state.qwen_endpoint or QWEN_ENDPOINT
        if not endpoint:
            raise RuntimeError("QWEN_ENDPOINT not configured")
        
        # Prepare the prompt
        system_prompt = "You are a helpful assistant that produces concise summaries."
        user_prompt = f"""Provide a {state.summary_type} summary for the following text.
Return concise bullet points in JSON format under keys: {{"type": "<type>", "bullets": ["..."]}}.

Text:
{state.text}"""
        
        # Make API call to Qwen
        headers = {
            "Content-Type": "application/json"
        }
        if QWEN_API_KEY:
            headers["Authorization"] = f"Bearer {QWEN_API_KEY}"
        
        payload = {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Qwen API error: {response.status_code} - {response.text}")
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Try to parse the JSON response
            try:
                summary_json = json.loads(content)
            except:
                # Fallback to raw content if not valid JSON
                summary_json = {
                    "type": state.summary_type,
                    "content": content
                }
            
            state.result = {
                "transcript": None,
                "summary": summary_json
            }
            return state
            
    except Exception as e:
        logger.error(f"Qwen summarization failed: {e}")
        state.error = str(e)
        return state

async def maybe_callback(state: WorkflowState) -> WorkflowState:
    """Send callback if URL is provided"""
    if state.callback_url and state.result:
        try:
            headers = {"Content-Type": "application/json"}
            if LANGGRAPH_CALLBACK_SECRET:
                headers["X-LangGraph-Secret"] = LANGGRAPH_CALLBACK_SECRET
            
            callback_data = {
                "job_id": state.job_id,
                "status": "completed" if not state.error else "failed",
                "result": state.result,
                "error": state.error
            }
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    state.callback_url,
                    json=callback_data,
                    headers=headers,
                    timeout=30
                )
            logger.info(f"Callback sent to {state.callback_url}")
        except Exception as e:
            logger.error(f"Failed to send callback: {e}")
    
    return state

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("route", route_by_action)
    workflow.add_node("transcribe_placeholder", transcribe_placeholder)
    workflow.add_node("summarize_prepare_text", summarize_prepare_text)
    workflow.add_node("summarize_qwen", summarize_qwen)
    workflow.add_node("maybe_callback", maybe_callback)
    
    # Add edges
    workflow.add_conditional_edges(
        "route",
        lambda x: x.action,
        {
            "transcribe": "transcribe_placeholder",
            "summarize": "summarize_prepare_text"
        }
    )
    
    workflow.add_edge("transcribe_placeholder", "maybe_callback")
    workflow.add_edge("summarize_prepare_text", "summarize_qwen")
    workflow.add_edge("summarize_qwen", "maybe_callback")
    workflow.add_edge("maybe_callback", END)
    
    # Set entry point
    workflow.set_entry_point("route")
    
    return workflow

# Create the app
checkpointer = MemorySaver()
workflow = create_workflow()
app = workflow.compile(checkpointer=checkpointer)

async def process_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a workflow request"""
    try:
        state = WorkflowState(**request_data)
        result = await app.ainvoke(state)
        
        return {
            "job_id": result.get("job_id"),
            "status": "completed" if not result.get("error") else "failed",
            "result": result.get("result"),
            "error": result.get("error")
        }
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            "job_id": request_data.get("job_id"),
            "status": "failed",
            "result": None,
            "error": str(e)
        }

if __name__ == "__main__":
    # Test the workflow locally
    async def test():
        # Test transcription
        transcribe_request = {
            "action": "transcribe",
            "job_id": "test-transcribe-001",
            "inputs": {
                "file_path": "/path/to/audio.wav",
                "diarize": False
            }
        }
        
        print("Testing transcription...")
        result = await process_request(transcribe_request)
        print(f"Transcription result: {json.dumps(result, indent=2)}")
        
        # Test summarization
        summarize_request = {
            "action": "summarize",
            "job_id": "test-summarize-001",
            "inputs": {
                "text": "This is a test text that needs to be summarized. It contains important information about testing.",
                "summary_type": "executive"
            },
            "qwen_endpoint": QWEN_ENDPOINT
        }
        
        print("\nTesting summarization...")
        result = await process_request(summarize_request)
        print(f"Summarization result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test())
