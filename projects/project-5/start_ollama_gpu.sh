#!/usr/bin/env bash

# Start Ollama server pinned to a specific GPU using CUDA/HIP visibility vars.
#
# Usage:
#   bash scripts/start_ollama_gpu.sh [--device N] [--background]
#   bash scripts/start_ollama_gpu.sh 1           # shorthand for --device 1
#   OLLAMA_CUDA_VISIBLE_DEVICES=1 bash scripts/start_ollama_gpu.sh
#
# Notes:
# - If --device N is provided, we export:
#     CUDA_VISIBLE_DEVICES=N (NVIDIA) and HIP_VISIBLE_DEVICES=N (AMD)
# - If OLLAMA_CUDA_VISIBLE_DEVICES/OLLAMA_HIP_VISIBLE_DEVICES are set in .env
#   or the environment, they are used unless overridden by --device.
# - Without any selection, Ollama will see all GPUs as usual.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${THIS_DIR}" && pwd)"

# Load project .env if present
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DEVICE_ARG=""
BACKGROUND=false

usage() {
  cat <<USAGE
Start Ollama pinned to a specific GPU using CUDA/HIP visibility variables.

Usage:
  bash scripts/start_ollama_gpu.sh [--device N] [--background]
  bash scripts/start_ollama_gpu.sh 1

Environment (from .env or shell):
  OLLAMA_CUDA_VISIBLE_DEVICES   # e.g., 1
  OLLAMA_HIP_VISIBLE_DEVICES    # e.g., 1

Examples:
  # Pin to GPU 1 (second GPU) and run in foreground
  bash scripts/start_ollama_gpu.sh --device 1

  # Same using shorthand and background mode
  bash scripts/start_ollama_gpu.sh 1 --background

  # Use values from .env
  bash scripts/start_ollama_gpu.sh
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0 ;;
    --device)
      shift; DEVICE_ARG="${1:-}" ;;
    --background)
      BACKGROUND=true ;;
    ''|*[!0-9]*)
      # Non-numeric token that isn't a known flag
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
    *)
      # Numeric position arg acts like --device N
      DEVICE_ARG="$1" ;;
  esac
  shift || true
done

# Determine which visibility variables to export
CUDA_VIS="${OLLAMA_CUDA_VISIBLE_DEVICES:-}"
HIP_VIS="${OLLAMA_HIP_VISIBLE_DEVICES:-}"

if [[ -n "${DEVICE_ARG}" ]]; then
  CUDA_VIS="${DEVICE_ARG}"
  HIP_VIS="${DEVICE_ARG}"
fi

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ollama_serve.log"

echo "[start_ollama_gpu] Preparing to start ollama serve"
if [[ -n "${CUDA_VIS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VIS}"
  echo "[start_ollama_gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  echo "[start_ollama_gpu] CUDA_VISIBLE_DEVICES not set (all NVIDIA GPUs visible)"
fi
if [[ -n "${HIP_VIS}" ]]; then
  export HIP_VISIBLE_DEVICES="${HIP_VIS}"
  echo "[start_ollama_gpu] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
else
  echo "[start_ollama_gpu] HIP_VISIBLE_DEVICES not set (all AMD GPUs visible)"
fi

echo "[start_ollama_gpu] Starting: ollama serve"
if [[ "${BACKGROUND}" == true ]]; then
  # Run in background with nohup; write stdout/stderr to log file
  nohup bash -lc 'ollama serve' >>"${LOG_FILE}" 2>&1 &
  PID=$!
  echo "[start_ollama_gpu] ollama serve started in background (PID ${PID}). Logs: ${LOG_FILE}"
else
  # Foreground; user can Ctrl-C to stop
  exec ollama serve
fi
