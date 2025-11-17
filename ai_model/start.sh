#!/usr/bin/env bash
# Render deployment startup script
# This script is used by Render to start the application

set -euo pipefail

echo "Starting Paddy Disease API..."

# Use PORT environment variable from Render, default to 8000 for local testing
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

# Use ONNX-based app for lightweight deployment
APP_MODULE=${APP_MODULE:-app_onnx:app}

echo "Starting uvicorn on $HOST:$PORT with module $APP_MODULE"

# Start the server
python3 -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
