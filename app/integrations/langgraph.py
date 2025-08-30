import os
import time
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .remote_client import RemoteClient

WORKFLOW_ID = os.getenv("LANGGRAPH_WORKFLOW_ID", "transcribe_summarize_v1")
TRIGGER_URL = os.getenv("LANGGRAPH_ENDPOINT")  # e.g., https://host/trigger
RESULT_URL = os.getenv("LANGGRAPH_RESULT_ENDPOINT")  # e.g., https://host/result (optional)
API_KEY = os.getenv("MODEL_API_KEY")
CALLBACK_URL = os.getenv("LANGGRAPH_CALLBACK_URL")
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT")
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT")


class LangGraphClient:
    def __init__(self):
        if not TRIGGER_URL:
            raise RuntimeError("LANGGRAPH_ENDPOINT is not configured")
        self.trigger_client = RemoteClient(TRIGGER_URL, api_key=API_KEY)
        self.result_client = RemoteClient(RESULT_URL, api_key=API_KEY) if RESULT_URL else None

    def _base_payload(self, action: str, inputs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        payload = {
            "workflow_id": WORKFLOW_ID,
            "action": action,
            "job_id": job_id,
            "inputs": inputs,
        }
        if CALLBACK_URL:
            payload["callback_url"] = CALLBACK_URL
        # Forward model endpoints if provided (so LangGraph can call them)
        if WHISPER_ENDPOINT:
            payload["whisper_endpoint"] = WHISPER_ENDPOINT
        if QWEN_ENDPOINT:
            payload["qwen_endpoint"] = QWEN_ENDPOINT
        return payload

    def trigger(self, action: str, inputs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        payload = self._base_payload(action, inputs, job_id)
        return self.trigger_client.post_json(json=payload)

    def poll_until_complete(
        self,
        run_ref: Dict[str, Any],
        poll_interval: float = 2.0,
        timeout_seconds: float = 300.0,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Poll result endpoint until status is 'completed' or 'failed'.
        Returns (result_json, error_message).
        If RESULT_URL is not configured, returns (None, "Result endpoint not configured").
        """
        if not self.result_client:
            return None, "Result endpoint not configured"

        # Accept run_ref as {'run_id': '...'} or {'job_id': '...'}
        params: Dict[str, Any] = {}
        if "run_id" in run_ref:
            params["run_id"] = run_ref["run_id"]
        if "job_id" in run_ref:
            params["job_id"] = run_ref["job_id"]

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            data = self.result_client.get_json(params=params)
            status = data.get("status") or data.get("state")
            if status in ("completed", "succeeded", "success"):
                return data, None
            if status in ("failed", "error"):
                return None, data.get("error") or "LangGraph reported failure"
            time.sleep(poll_interval)
        return None, "Timed out waiting for LangGraph result"


def get_client() -> LangGraphClient:
    return LangGraphClient()
