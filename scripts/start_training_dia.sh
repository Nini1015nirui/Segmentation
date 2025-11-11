#!/usr/bin/env bash
set -eo pipefail

# Attempt to activate conda 'seg' env if available (non-fatal)
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate seg >/dev/null 2>&1 || true
fi
set -u

# Resolve project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# nnUNet data paths rooted at project
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Prefer seg environment python, fallback to system python
RUN_PY="/root/miniconda3/envs/seg/bin/python"
if [ ! -x "$RUN_PY" ]; then
  RUN_PY="python"
fi

# Ensure local lightm-unet is importable (no need to pip install -e)
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"

# Resource/memory knobs
export nnUNet_n_proc_DA="4"
export CUDA_LAUNCH_BLOCKING="1"
# native allocator + conservative split to avoid fragmentation on 8GB GPUs
export PYTORCH_CUDA_ALLOC_CONF="backend:native,expandable_segments:True,max_split_size_mb:64"
export NNUNET_USE_LCC_POSTPROC="1"
# enforce a conservative batch size unless user overrides
export NNUNET_BATCH_SIZE=${NNUNET_BATCH_SIZE:-2}

# Dataset config (DIA)
DATASET_ID="804"
DATASET_NAME="DIA"
PLANS_ID="nnUNetPlans_mem4"
CHECKPOINT_DIR="$nnUNet_results/Dataset${DATASET_ID}_$DATASET_NAME/nnUNetTrainerLightMUNet__nnUNetPlans__2d/fold_all"

# Summary of environment
echo "🔧 训练环境摘要"
"$RUN_PY" - << 'PY'
import torch, os
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print('nnUNet_raw:', os.environ.get('nnUNet_raw'))
print('nnUNet_preprocessed:', os.environ.get('nnUNet_preprocessed'))
print('nnUNet_results:', os.environ.get('nnUNet_results'))
PY

# Stop stale DIA trainings (if any)
echo "🧹 终止可能存在的旧训练进程..."
pkill -f "nnunetv2.run.run_training ${DATASET_ID} 2d all -tr nnUNetTrainerLightMUNet" || true
sleep 2

# Clean existing DIA checkpoints for a truly fresh run
if [ -d "$CHECKPOINT_DIR" ]; then
  echo "🗑️ 清理旧的检查点: $CHECKPOINT_DIR"
  rm -rf "$CHECKPOINT_DIR"/*
fi

# Launch full fresh training
LOG_OUT="/tmp/nnunet_${DATASET_NAME}_training_$(date +%Y%m%d_%H%M%S).out"
echo "🚀 启动完整 2D 训练 (Dataset${DATASET_ID}_${DATASET_NAME}, plans=${PLANS_ID})..."
echo "   日志: $LOG_OUT"

# Prefer CUDA when available, otherwise CPU
DEVICE_OPT=$("$RUN_PY" - << 'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)

CUDA_VISIBLE_DEVICES=0 nohup "$RUN_PY" -m nnunetv2.run.run_training \
  "${DATASET_ID}" 2d all -tr nnUNetTrainerLightMUNet -p "${PLANS_ID}" -device "$DEVICE_OPT" \
  >> "$LOG_OUT" 2>&1 &
PID=$!

echo "✅ 训练已启动 (PID=$PID)"
echo "📂 训练目录: $CHECKPOINT_DIR"
echo "📊 实时查看: tail -f $LOG_OUT"
echo "🛑 停止训练: kill $PID"
