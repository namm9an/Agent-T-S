#!/usr/bin/env python3
"""
LangGraph HTTP Server for local development
Provides /trigger and /result endpoints compatible with the FastAPI integration
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from langgraph_app import process_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for LangGraph
app = FastAPI(title="LangGraph Server", version="1.0.0")

# In-memory storage for job results (for development only)
job_results: Dict[str, Dict[str, Any]] = {}

class TriggerRequest(BaseModel):
    workflow_id: str
    action: str
    job_id: str
    inputs: Dict[str, Any]
    callback_url: Optional[str] = None
    qwen_endpoint: Optional[str] = None
    whisper_endpoint: Optional[str] = None

class ResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "langgraph",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/trigger")
async def trigger_workflow(request: TriggerRequest):
    """
    Trigger a workflow execution
    Compatible with the FastAPI integration's expectations
    """
    try:
        logger.info(f"Triggering workflow: {request.workflow_id} for job {request.job_id}")
        
        # Create workflow request
        workflow_request = {
            "action": request.action,
            "job_id": request.job_id,
            "inputs": request.inputs,
            "callback_url": request.callback_url,
            "qwen_endpoint": request.qwen_endpoint,
            "whisper_endpoint": request.whisper_endpoint
        }
        
        # Process asynchronously
        asyncio.create_task(execute_workflow(workflow_request))
        
        # Return immediate response
        return {
            "job_id": request.job_id,
            "run_id": str(uuid4()),  # Generate a run ID
            "status": "queued",
            "message": "Workflow triggered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_workflow(workflow_request: Dict[str, Any]):
    """Execute workflow and store result"""
    try:
        result = await process_request(workflow_request)
        job_id = workflow_request["job_id"]
        
        # Store result for polling
        job_results[job_id] = {
            "job_id": job_id,
            "status": result.get("status", "completed"),
            "result": result.get("result"),
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Workflow completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        job_id = workflow_request.get("job_id")
        if job_id:
            job_results[job_id] = {
                "job_id": job_id,
                "status": "failed",
                "result": None,
                "error": {"message": str(e)},
                "timestamp": datetime.now().isoformat()
            }

@app.get("/result")
async def get_result(job_id: Optional[str] = None, run_id: Optional[str] = None):
    """
    Get workflow result by job_id or run_id
    For simplicity, we only use job_id in this implementation
    """
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id parameter required")
    
    if job_id not in job_results:
        # Return pending status if result not yet available
        return {
            "job_id": job_id,
            "status": "running",
            "result": None,
            "error": None
        }
    
    return job_results[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "jobs": list(job_results.keys()),
        "count": len(job_results)
    }

@app.delete("/jobs/{job_id}")
async def clear_job(job_id: str):
    """Clear a specific job result (for debugging)"""
    if job_id in job_results:
        del job_results[job_id]
        return {"message": f"Job {job_id} cleared"}
    raise HTTPException(status_code=404, detail="Job not found")

@app.delete("/jobs")
async def clear_all_jobs():
    """Clear all job results (for debugging)"""
    count = len(job_results)
    job_results.clear()
    return {"message": f"Cleared {count} jobs"}

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("LANGGRAPH_PORT", "8123"))
    host = os.getenv("LANGGRAPH_HOST", "0.0.0.0")
    
    logger.info(f"Starting LangGraph server on {host}:{port}")
    logger.info("Endpoints:")
    logger.info(f"  - Trigger: http://{host}:{port}/trigger")
    logger.info(f"  - Result:  http://{host}:{port}/result")
    logger.info(f"  - Health:  http://{host}:{port}/health")
    
    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")
