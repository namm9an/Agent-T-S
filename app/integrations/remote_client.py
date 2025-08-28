import os
from typing import Any, Dict, Optional

import httpx

DEFAULT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))


class RemoteClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=self.timeout, headers=headers)

    def post_json(self, url: Optional[str] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        target = (url or self.base_url)
        resp = self._client.post(target, json=json or {})
        resp.raise_for_status()
        if resp.text:
            return resp.json()
        return {}

    def get_json(self, url: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        target = (url or self.base_url)
        resp = self._client.get(target, params=params or {})
        resp.raise_for_status()
        if resp.text:
            return resp.json()
        return {}

    def close(self):
        self._client.close()
