#!/bin/bash

# Kill everything
killall python3 2>/dev/null
killall streamlit 2>/dev/null
sleep 1

# Activate virtual environment
source .venv/bin/activate

# Start backend
python3 app_e2e.py &
BACKEND_PID=$!
echo "Backend started: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend
streamlit run streamlit_app.py --server.headless true --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "Frontend started: $FRONTEND_PID"

echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:8501"
echo ""
echo "PIDs: Backend=$BACKEND_PID Frontend=$FRONTEND_PID"
echo "To kill: kill $BACKEND_PID $FRONTEND_PID"
