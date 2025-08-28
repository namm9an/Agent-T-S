# Usage: python3 scripts/test_qwen.py
# Requires env:
#   E2E_QWEN_BASE_URL (e.g., https://infer.e2enetworks.net/project/p-6530/endpoint/is-6346/v1/)
#   E2E_QWEN_TOKEN (Bearer token for the endpoint)

import os
import sys

BASE_URL = os.getenv("E2E_QWEN_BASE_URL")
TOKEN = os.getenv("E2E_QWEN_TOKEN")

if not BASE_URL or not TOKEN:
    print("Missing E2E_QWEN_BASE_URL or E2E_QWEN_TOKEN env vars", file=sys.stderr)
    sys.exit(2)

try:
    # Prefer new OpenAI v1 style client
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=TOKEN, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[{"role": "user", "content": "Create a program for checking whether a number is Prime or not?"}],
    )
    print(resp.choices[0].message.content)
except Exception:
    # Fallback to module-level usage if necessary
    import openai  # type: ignore

    openai.api_key = TOKEN
    openai.base_url = BASE_URL
    resp = openai.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[{"role": "user", "content": "Create a program for checking whether a number is Prime or not?"}],
    )
    print(resp.choices[0].message.content)

