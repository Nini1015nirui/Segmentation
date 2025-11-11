#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run a Python (or bash) script with WSL CUDA libs on PATH.
# Usage: ./scripts/gpu_run.sh python scripts/gpu_sanity.py

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

exec "$@"

