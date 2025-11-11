#!/usr/bin/env bash
set -eo pipefail

# Try to activate conda 'seg' (non-fatal)
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate seg >/dev/null 2>&1 || true
fi
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
RUN_PY="/root/miniconda3/envs/seg/bin/python"; [ -x "$RUN_PY" ] || RUN_PY="python"

# Configurable
DATASET_ID="${DATASET_ID:-804}"
DATASET_NAME="${DATASET_NAME:-}"
CONFIG="${CONFIG:-2d}"
PLANS_ID_USER="${PLANS_ID:-}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native,expandable_segments:True,max_split_size_mb:64}"
export NNUNET_BATCH_SIZE="${NNUNET_BATCH_SIZE:-2}"

# Detect dataset name
if [ -z "$DATASET_NAME" ]; then
  cand=$(ls -d "$nnUNet_preprocessed"/Dataset${DATASET_ID}_* 2>/dev/null | head -n1 || true)
  if [ -n "$cand" ]; then
    bn=$(basename "$cand")
    DATASET_NAME="${bn#Dataset${DATASET_ID}_}"
  else
    for nm in DIA TA; do
      if [ -d "$nnUNet_raw/Dataset${DATASET_ID}_$nm" ] || [ -d "$nnUNet_preprocessed/Dataset${DATASET_ID}_$nm" ]; then
        DATASET_NAME="$nm"; break
      fi
    done
    [ -z "$DATASET_NAME" ] && DATASET_NAME="DIA"
  fi
fi

# Plans id
PLANS_ID="$PLANS_ID_USER"
if [ -z "$PLANS_ID" ]; then
  if [ -f "$nnUNet_preprocessed/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetPlans_mem4.json" ]; then
    PLANS_ID="nnUNetPlans_mem4"
  else
    PLANS_ID="nnUNetPlans"
  fi
fi

CHECKPOINT_DIR="$nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainerLightMUNet__${PLANS_ID}__${CONFIG}/fold_all"
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "❌ 检查点目录不存在: $CHECKPOINT_DIR"
  echo "   请先运行 start_training.sh 开始训练"
  exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/checkpoint_best.pth" ] && [ ! -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
  echo "❌ 未找到检查点文件！"
  echo "   检查点目录: $CHECKPOINT_DIR"
  echo "   请先运行 start_training.sh 开始训练"
  exit 1
fi

# Clear GPU cache
$RUN_PY - <<'PY'
import torch
torch.cuda.empty_cache() if torch.cuda.is_available() else None
PY

DEVICE_OPT=$($RUN_PY - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)

echo "🔄 继续训练: Dataset${DATASET_ID}_${DATASET_NAME}, ${CONFIG}, plans=${PLANS_ID}"
echo "   检查点目录: $CHECKPOINT_DIR"

# Show brief summary
$RUN_PY - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
PY

# Stop stale
pkill -f "nnunetv2.run.run_training ${DATASET_ID} ${CONFIG} all -tr nnUNetTrainerLightMUNet" || true
sleep 1

LOG_OUT="/tmp/nnunet_resume_training_$(date +%Y%m%d_%H%M%S).out"
CUDA_VISIBLE_DEVICES=0 nohup "$RUN_PY" -m nnunetv2.run.run_training \
  "${DATASET_ID}" "${CONFIG}" all -tr nnUNetTrainerLightMUNet -p "${PLANS_ID}" --c -device "$DEVICE_OPT" \
  >> "$LOG_OUT" 2>&1 &
PID=$!
echo "✅ 训练已恢复 PID=$PID"
echo "📝 日志: $LOG_OUT"
echo "📂 目录: $CHECKPOINT_DIR"
