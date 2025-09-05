#!/bin/bash

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "python3 app_e2e.py" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# Activate virtual environment
source .venv/bin/activate

echo "Starting backend server on port 8000..."
python3 app_e2e.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

echo "Starting Streamlit frontend on port 8501..."
streamlit run streamlit_app.py --server.headless true --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "========================================="
echo "Services are starting up..."
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:8501"
echo ""
echo "To stop services, run:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "Or press Ctrl+C"
echo "========================================="

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
