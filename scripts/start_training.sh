#!/usr/bin/env bash
set -eo pipefail

# Try to activate conda 'seg' (non-fatal)
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate seg >/dev/null 2>&1 || true
fi
set -u

# Paths relative to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Ensure local nnUNetv2 package is importable
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
RUN_PY="/root/miniconda3/envs/seg/bin/python"; [ -x "$RUN_PY" ] || RUN_PY="python"

# Configurable knobs (env-overridable)
DATASET_ID="${DATASET_ID:-804}"            # default to DIA
DATASET_NAME="${DATASET_NAME:-}"            # auto-detect if empty
CONFIG="${CONFIG:-2d}"
PLANS_ID_USER="${PLANS_ID:-}"                # if empty, auto-choose
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native,expandable_segments:True,max_split_size_mb:64}"
export NNUNET_USE_LCC_POSTPROC="${NNUNET_USE_LCC_POSTPROC:-1}"
export NNUNET_BATCH_SIZE="${NNUNET_BATCH_SIZE:-2}"

# Auto-detect dataset name if not provided
if [ -z "$DATASET_NAME" ]; then
  cand=$(ls -d "$nnUNet_preprocessed"/Dataset${DATASET_ID}_* 2>/dev/null | head -n1 || true)
  if [ -n "$cand" ]; then
    bn=$(basename "$cand")
    DATASET_NAME="${bn#Dataset${DATASET_ID}_}"
  else
    # fallback common names
    for nm in DIA TA; do
      if [ -d "$nnUNet_raw/Dataset${DATASET_ID}_$nm" ] || [ -d "$nnUNet_preprocessed/Dataset${DATASET_ID}_$nm" ]; then
        DATASET_NAME="$nm"; break
      fi
    done
    [ -z "$DATASET_NAME" ] && DATASET_NAME="DIA"
  fi
fi

# Auto-pick plans id (prefer mem4 if present)
PLANS_ID="$PLANS_ID_USER"
if [ -z "$PLANS_ID" ]; then
  if [ -f "$nnUNet_preprocessed/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetPlans_mem4.json" ]; then
    PLANS_ID="nnUNetPlans_mem4"
  else
    PLANS_ID="nnUNetPlans"
  fi
fi

# Device choose
DEVICE_OPT=$($RUN_PY - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)

# Summary
echo "🔧 全新训练配置"
echo "   Dataset: Dataset${DATASET_ID}_${DATASET_NAME}"
echo "   Config:  ${CONFIG}"
echo "   Plans:   ${PLANS_ID}"
echo "   Batch:   ${NNUNET_BATCH_SIZE}"
echo "   Device:  ${DEVICE_OPT}"

# Stop stale trainings
echo "🧹 Cleaning up old processes..."
pkill -f "nnunetv2.run.run_training ${DATASET_ID} ${CONFIG} all -tr nnUNetTrainerLightMUNet" || true
sleep 1

# Clear GPU cache
$RUN_PY - <<'PY'
import torch
torch.cuda.empty_cache() if torch.cuda.is_available() else None
PY

# Clean checkpoints
CHECKPOINT_DIR="$nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainerLightMUNet__${PLANS_ID}__${CONFIG}/fold_all"
if [ -d "$CHECKPOINT_DIR" ]; then
  echo "🗑️ 清理旧检查点: $CHECKPOINT_DIR"
  rm -rf "$CHECKPOINT_DIR"/*
fi

# Launch
LOG_OUT="/tmp/nnunet_${DATASET_NAME}_training_$(date +%Y%m%d_%H%M%S).out"
echo "🚀 启动训练: Dataset${DATASET_ID}_${DATASET_NAME}, ${CONFIG}, plans=${PLANS_ID}"
CUDA_VISIBLE_DEVICES=0 nohup "$RUN_PY" -m nnunetv2.run.run_training \
  "${DATASET_ID}" "${CONFIG}" all -tr nnUNetTrainerLightMUNet -p "${PLANS_ID}" -device "$DEVICE_OPT" \
  >> "$LOG_OUT" 2>&1 &
PID=$!
echo "✅ 训练已启动 PID=$PID"
echo "📝 日志: $LOG_OUT"
echo "📂 目录: $CHECKPOINT_DIR"
echo "📊 监控: tail -f $LOG_OUT"
