#!/usr/bin/env bash
set -eo pipefail

# Attempt to activate conda 'seg' env if available (non-fatal)
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

# Prefer seg environment python, fallback to system python
RUN_PY="/root/miniconda3/envs/seg/bin/python"
if [ ! -x "$RUN_PY" ]; then
  RUN_PY="python"
fi

# Ensure local lightm-unet is importable
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"

DATASET_ID="804"
echo "🔎 预处理 DIA 数据集 Dataset${DATASET_ID}_DIA（内存目标：4GB，自定义计划名，仅2D）"

# 1) Fingerprint with integrity check
"$RUN_PY" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d "$DATASET_ID" --verify_dataset_integrity --clean --no_pp

# 2) Plan experiments with reduced GPU memory target and custom plans name
"$RUN_PY" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d "$DATASET_ID" -gpu_memory_target 4 -overwrite_plans_name nnUNetPlans_mem4 --no_pp

# 3) Preprocess only 2D with the custom plans id
"$RUN_PY" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d "$DATASET_ID" -pl ExperimentPlanner -gpu_memory_target 4 -overwrite_plans_name nnUNetPlans_mem4

echo "✅ 预处理完成: $nnUNet_preprocessed/Dataset${DATASET_ID}_DIA (plans: nnUNetPlans_mem4, 2D)"
