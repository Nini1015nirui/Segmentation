#!/usr/bin/env bash
set -euo pipefail

# Logging
MON_LOG="/tmp/dia_monitor_$(date +%Y%m%d_%H%M%S).out"
exec > >(tee -a "$MON_LOG") 2>&1

echo "[MON] DIA training monitor starting at $(date)"

# Try conda seg (non-fatal)
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate seg >/dev/null 2>&1 || true
fi
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# GPU/alloc knobs
export PYTORCH_CUDA_ALLOC_CONF="backend:native,expandable_segments:True,max_split_size_mb:64"
export NNUNET_USE_LCC_POSTPROC="1"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

DSID=804
PLANS_ID="nnUNetPlans_mem4"

echo "[MON] Checking preprocess/plans for Dataset${DSID}_DIA..."
if [ ! -f "$nnUNet_preprocessed/Dataset${DSID}_DIA/${PLANS_ID}.json" ]; then
  echo "[MON] Missing ${PLANS_ID}.json -> run low-mem 2D preprocess"
  /root/miniconda3/envs/seg/bin/python "$SCRIPT_DIR/preprocess_2d_mem4.py"
  echo "[MON] Preprocess finished at $(date)"
else
  echo "[MON] Found ${PLANS_ID}.json, skip preprocess"
fi

start_training() {
  local bs="$1"
  echo "[MON] Starting training with NNUNET_BATCH_SIZE=$bs ..."
  export NNUNET_BATCH_SIZE="$bs"
  # clean any stale processes
  pkill -f "nnUNetv2_train ${DSID}" || true
  pkill -f "nnunetv2.run.run_training ${DSID}" || true
  sleep 1
  # launch
  /bin/bash "$SCRIPT_DIR/start_training_dia.sh" | tee /tmp/.dia_start.out
  LOG=$(rg -n "日志: /tmp/nnunet_DIA_training_.*.out" -N /tmp/.dia_start.out | sed -E 's/.*(\/tmp\/nnunet_DIA_training_[^ ]+\.out).*/\1/' | tail -n1)
  PID=$(rg -n "PID=([0-9]+)" -N /tmp/.dia_start.out | sed -E 's/.*PID=([0-9]+).*/\1/' | tail -n1)
  if [ -z "$LOG" ]; then LOG=$(ls -1 /tmp/nnunet_DIA_training_*.out 2>/dev/null | tail -n1 || true); fi
  echo "[MON] LOG=$LOG PID=$PID"
}

monitor_loop() {
  local pid="$1"; local log="$2"; local bs="$3"
  echo "[MON] Enter monitor loop. PID=$pid LOG=$log"
  while kill -0 "$pid" >/dev/null 2>&1; do
    echo "--- tail ($log) @ $(date) ---"; tail -n 60 "$log" || true
    if rg -q "out of memory|CUDA out of memory" "$log"; then
      echo "[MON] Detected OOM at $(date). Will restart with smaller batch."
      kill "$pid" 2>/dev/null || true
      sleep 3
      if [ "$bs" -gt 1 ]; then
        start_training 1
        pid="$PID"; log="$LOG"; bs=1
        echo "[MON] Restarted with NNUNET_BATCH_SIZE=1. New PID=$pid LOG=$log"
        continue
      else
        echo "[MON] Already at batch=1 but OOM persists. Consider reducing model width or patch size."
        break
      fi
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "--- nvidia-smi @ $(date) ---"
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
    sleep 60
  done
  echo "[MON] Training process ended at $(date).";
}

start_training 2
monitor_loop "$PID" "$LOG" 2

echo "[MON] Monitor finished. See $MON_LOG for full history."

