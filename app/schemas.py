from enum import Enum
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Timestamps(BaseModel):
    created_at: str
    updated_at: str


class JobResult(BaseModel):
    transcript: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


class Job(BaseModel):
    id: str
    type: str
    status: JobStatus
    inputs: Dict[str, Any]
    result: JobResult
    error: Optional[Any] = None
    timestamps: Timestamps


class EnqueueResponse(BaseModel):
    id: str
    status: JobStatus = Field(default=JobStatus.queued)


class JobResponse(BaseModel):
    job: Job


class SummarizeRequest(BaseModel):
    job_id: Optional[str] = Field(default=None, description="Existing transcription job ID")
    text: Optional[str] = Field(default=None, description="Raw text to summarize if no job_id provided")
    summary_type: str = Field(default="executive")

    @model_validator(mode="after")
    def validate_one_of(cls, values):  # type: ignore[override]
        # Ensure exactly one of job_id or text is provided
        if bool(values.job_id) == bool(values.text):  # both set or both None
            raise ValueError("Provide exactly one of 'job_id' or 'text'")
        return values


class LangGraphCallback(BaseModel):
    job_id: str
    status: Literal["completed", "failed"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Any] = None
