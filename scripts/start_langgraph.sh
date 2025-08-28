#!/usr/bin/env bash
# Start LangGraph Server for local development

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

# Set LangGraph server port
export LANGGRAPH_PORT="${LANGGRAPH_PORT:-8123}"
export LANGGRAPH_HOST="${LANGGRAPH_HOST:-0.0.0.0}"

echo "========================================="
echo "Starting LangGraph Server"
echo "========================================="
echo "Host: $LANGGRAPH_HOST"
echo "Port: $LANGGRAPH_PORT"
echo "Endpoints:"
echo "  - Trigger: http://localhost:$LANGGRAPH_PORT/trigger"
echo "  - Result:  http://localhost:$LANGGRAPH_PORT/result"
echo "  - Health:  http://localhost:$LANGGRAPH_PORT/health"
echo "========================================="

# Start the LangGraph server
python langgraph_server.py
