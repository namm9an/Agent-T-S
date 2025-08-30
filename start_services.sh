#!/bin/bash

echo "🚀 Starting Audio Transcription Services..."

# Kill any existing processes
echo "Cleaning up old processes..."
killall python3 2>/dev/null
killall Python 2>/dev/null
killall streamlit 2>/dev/null

# Start backend
echo "Starting backend API server..."
mkdir -p data/logs
nohup python3 app_e2e.py > data/logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Backend is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit
echo "Starting Streamlit UI..."
nohup streamlit run streamlit_app.py --server.headless true --server.port 8501 > data/logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Wait for Streamlit to be ready
echo "Waiting for Streamlit to be ready..."
sleep 3

# Check if both are running
if ps -p $BACKEND_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo ""
    echo "✅ All services started successfully!"
    echo ""
    echo "📍 Backend API: http://localhost:8000"
    echo "📍 Streamlit UI: http://localhost:8501"
    echo ""
    echo "📝 Logs:"
    echo "   - Backend: data/logs/backend.log"
    echo "   - Frontend: data/logs/frontend.log"
    echo ""
    echo "To stop services, run: killall python3 streamlit"
else
    echo "❌ Failed to start services. Check the logs."
    exit 1
fi
