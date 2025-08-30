#!/usr/bin/env bash
# Start Integrated LangGraph AI Agent Service

set -euo pipefail

# Change to project root
cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found!"
    exit 1
fi

# Create required directories
mkdir -p jobs uploads

echo "========================================="
echo "Starting LangGraph AI Agent Service"
echo "========================================="
echo "Service: Integrated (FastAPI + LangGraph)"
echo "Port: 8000"
echo "Swagger UI: http://localhost:8000/docs"
echo "========================================="
echo "Agents:"
echo "  - Whisper: ${WHISPER_ENDPOINT}"
echo "  - Qwen: ${QWEN_ENDPOINT}"
echo "========================================="

# Start the integrated service
python3 app_integrated.py
