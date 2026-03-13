#!/usr/bin/env bash

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PID_FILE=".server.pid"

stopped=false

# Stop by PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID $PID)..."
        kill "$PID" 2>/dev/null
        stopped=true
    fi
    rm -f "$PID_FILE"
fi

# Also kill anything on the port
EXISTING=$(lsof -i :"$PORT" -t 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Killing process on port $PORT (PID $EXISTING)..."
    kill "$EXISTING" 2>/dev/null || true
    stopped=true
fi

if [ "$stopped" = true ]; then
    echo "Server stopped."
else
    echo "No server running."
fi
