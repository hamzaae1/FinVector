#!/bin/bash

# FinVector - Launch Script
# Budget-Aware E-Commerce Search Engine

cd "$(dirname "$0")"

echo "=============================================="
echo "  FinVector - Budget-Aware Product Search"
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Check if activation worked
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment: $VIRTUAL_ENV"

# Check for .env file
if [ ! -f .env ]; then
    echo "WARNING: .env file not found"
fi

# Default port
PORT=${1:-8000}

echo "Starting API server on http://localhost:$PORT"
echo "API Docs: http://localhost:$PORT/docs"
echo "Press Ctrl+C to stop"
echo "=============================================="

# Launch the API
uvicorn api:app --host 0.0.0.0 --port $PORT --reload
