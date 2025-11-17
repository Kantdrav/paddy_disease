#!/usr/bin/env bash
# Helper to run the API using the project virtualenv
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Virtualenv python not found at $VENV_PY"
  echo "Create a venv and install requirements first: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi
# Activate for interactive shells (optional)
# shellcheck source=/dev/null
source "$ROOT_DIR/venv/bin/activate"

# Sanity check: required runtime packages
NEED_PKGS=()
if ! "$VENV_PY" -c 'import fastapi' >/dev/null 2>&1; then NEED_PKGS+=(fastapi); fi
if ! "$VENV_PY" -c 'import PIL' >/dev/null 2>&1; then NEED_PKGS+=(pillow); fi
if ! "$VENV_PY" -c 'import uvicorn' >/dev/null 2>&1; then NEED_PKGS+=(uvicorn); fi
if ! "$VENV_PY" -c 'import multipart' >/dev/null 2>&1; then NEED_PKGS+=(python-multipart); fi
if [ ${#NEED_PKGS[@]} -gt 0 ]; then
  echo "Missing runtime packages in venv: ${NEED_PKGS[*]}"
  echo "Installing now..."
  pip install "${NEED_PKGS[@]}"
fi

# Torch/torchvision are platform-specific; check and guide if missing
if ! "$VENV_PY" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PyTorch and/or torchvision are not installed in the venv."
  echo "Install CPU-only versions (example):"
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
  echo "Or install CUDA builds matching your system from: https://pytorch.org/get-started/locally/"
  exit 1
fi

# Allow overriding host/port for platforms like Render
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

# Default to original app; set APP_MODULE=app_tflite:app to run the TFLite API
APP_MODULE=${APP_MODULE:-app:app}

"$VENV_PY" -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
