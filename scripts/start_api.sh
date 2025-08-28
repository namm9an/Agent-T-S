#!/usr/bin/env bash
# Start FastAPI Application Server

set -euo pipefail

# Change to project root
cd "$(dirname "$0")/.."

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found. Using default values."
fi

# Create required directories
mkdir -p jobs uploads

echo "========================================="
echo "Starting FastAPI Application"
echo "========================================="
echo "Host: 0.0.0.0"
echo "Port: 8000"
echo "Swagger UI: http://localhost:8000/docs"
echo "========================================="
echo "Configuration:"
echo "  LangGraph: ${LANGGRAPH_ENDPOINT:-Not configured}"
echo "  Qwen: ${QWEN_ENDPOINT:-Not configured}"
echo "  Whisper: ${WHISPER_ENDPOINT:-Not configured}"
echo "========================================="

# Start FastAPI with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
