#!/bin/bash
# Stop the Living Portrait server gracefully

echo "Stopping Living Portrait server..."

# Try graceful shutdown via API first
curl -s http://localhost:8000/stop_server || true

# Wait a moment
sleep 2

# If still running, force kill
if lsof -ti :8000 > /dev/null 2>&1; then
    echo "Server still running, force stopping..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null
    echo "Server force stopped"
else
    echo "Server stopped gracefully"
fi
