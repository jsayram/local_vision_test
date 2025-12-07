#!/bin/bash
# Start the Living Portrait server with proper signal handling

cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Change to vision_experiment directory
cd vision_experiment

# Function to handle cleanup on Ctrl+C
cleanup() {
    echo ""
    echo "Ctrl+C detected, shutting down server..."
    curl -s http://localhost:8000/stop_server > /dev/null 2>&1 || true
    sleep 2
    
    # Force kill if still running
    if lsof -ti :8000 > /dev/null 2>&1; then
        lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    fi
    
    exit 0
}

# Set trap for Ctrl+C
trap cleanup INT TERM

echo "Starting Living Portrait server..."
echo "Press Ctrl+C to stop"
echo ""

# Start the Python app in foreground
python app.py

# If Python exits normally, cleanup
cleanup
