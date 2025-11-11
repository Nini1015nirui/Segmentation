#!/usr/bin/env bash
set -eo pipefail

# Purpose: Enable TTA (mirroring) at inference and auto-determine/apply nnU-Net postprocessing
# Usage examples:
#   bash scripts/apply_tta_and_postprocessing.sh 803 2d nnUNetTrainerLightMUNet
#   bash scripts/apply_tta_and_postprocessing.sh 803 3d_fullres nnUNetTrainerLightMUNet

DATASET_ID="${1:-803}"
CONFIGURATION="${2:-2d}"
TRAINER="${3:-nnUNetTrainerLightMUNet}"
PLANS="${4:-}"
NUM_PROCS="${NNUNET_NP:-4}"
SAMPLE_COUNT="${NNUNET_SAMPLE_COUNT:-6}"

# Try to activate conda env if available
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
fi
conda activate seg >/dev/null 2>&1 || true
# prefer seg env console scripts
export PATH="/root/miniconda3/envs/seg/bin:${PATH}"
# tame OpenMP/BLAS threading and SHM requirements
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
set -u

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_DIR/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_DIR/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_DIR/data/nnUNet_results}"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"

# Resolve dataset name (e.g., Dataset804_DIA) from id if present
DATASET_DIR_CAND=$(ls -d "$nnUNet_raw"/Dataset${DATASET_ID}_* 2>/dev/null | head -n1 || true)
if [[ -n "$DATASET_DIR_CAND" ]]; then
  DATASET_NAME_OR_ID="$(basename "$DATASET_DIR_CAND")"
else
  DATASET_NAME_OR_ID="${DATASET_ID}"
fi
# auto-detect plans if not provided (prefer mem4)
if [[ -z "$PLANS" ]]; then
  if ls "$nnUNet_preprocessed"/Dataset${DATASET_ID}_*/nnUNetPlans_mem4.json >/dev/null 2>&1; then
    PLANS="nnUNetPlans_mem4"
  else
    PLANS="nnUNetPlans"
  fi
fi

MODEL_DIR="$nnUNet_results/Dataset${DATASET_ID}_*/${TRAINER}__${PLANS}__${CONFIGURATION}"
MODEL_DIR_RESOLVED=$(ls -d $MODEL_DIR 2>/dev/null | head -n1 || true)
if [[ -z "$MODEL_DIR_RESOLVED" ]]; then
  echo "❌ 未找到模型目录: $MODEL_DIR"
  echo "   请确认已完成训练，或手动指定路径。"
  exit 1
fi

FOLD_DIR="$MODEL_DIR_RESOLVED/fold_all"
PLANS_JSON="$MODEL_DIR_RESOLVED/plans.json"
DATASET_JSON="$MODEL_DIR_RESOLVED/dataset.json"

LABELS_TR="$nnUNet_raw/Dataset${DATASET_ID}_*/labelsTr"
LABELS_TR_RESOLVED=$(ls -d $LABELS_TR 2>/dev/null | head -n1 || true)
if [[ -z "$LABELS_TR_RESOLVED" ]]; then
  echo "❌ 未找到 labelsTr 目录: $LABELS_TR"
  exit 1
fi

# resolve imagesTr dir for PP determination
IMAGES_TR_DIR="$nnUNet_raw/Dataset${DATASET_ID}_*/imagesTr"
IMAGES_TR_RESOLVED=$(ls -d $IMAGES_TR_DIR 2>/dev/null | head -n1 || true)
if [[ -z "$IMAGES_TR_RESOLVED" ]]; then
  echo "❌ 未找到 imagesTr 目录: $IMAGES_TR_DIR"
  exit 1
fi

IMAGES_TS="$nnUNet_raw/Dataset${DATASET_ID}_*/imagesTs"
IMAGES_TS_RESOLVED=$(ls -d $IMAGES_TS 2>/dev/null | head -n1 || true)
if [[ -z "$IMAGES_TS_RESOLVED" ]]; then
  echo "⚠️ 未找到 imagesTs，将使用 imagesTr 作为示例"
  IMAGES_TS_RESOLVED=$(ls -d $nnUNet_raw/Dataset${DATASET_ID}_*/imagesTr 2>/dev/null | head -n1 || true)
fi
# If imagesTs exists but is empty, also fallback to imagesTr
if [[ -d "$IMAGES_TS_RESOLVED" ]] && [[ -z "$(ls -A "$IMAGES_TS_RESOLVED" 2>/dev/null)" ]]; then
  echo "⚠️ imagesTs 目录为空，将使用 imagesTr 作为示例"
  IMAGES_TS_RESOLVED="$IMAGES_TR_RESOLVED"
fi

# build small sample folders for quick demo
SAMPLE_TR_DIR="$PROJECT_DIR/data/samples/pp_tr_${DATASET_ID}_${CONFIGURATION}"
SAMPLE_TS_DIR="$PROJECT_DIR/data/samples/pp_ts_${DATASET_ID}_${CONFIGURATION}"
mkdir -p "$SAMPLE_TR_DIR" "$SAMPLE_TS_DIR"
# Avoid literal globs when empty
shopt -s nullglob
# Copy limited number of files without piping to head to avoid SIGPIPE under 'set -o pipefail'
cnt=0
for f in "$IMAGES_TR_RESOLVED"/*; do
  base="$(basename "$f")"
  cp -f "$f" "$SAMPLE_TR_DIR/$base"
  cnt=$((cnt+1))
  if [ "$cnt" -ge "$SAMPLE_COUNT" ]; then break; fi
done

cnt=0
for f in "$IMAGES_TS_RESOLVED"/*; do
  base="$(basename "$f")"
  cp -f "$f" "$SAMPLE_TS_DIR/$base"
  cnt=$((cnt+1))
  if [ "$cnt" -ge "$SAMPLE_COUNT" ]; then break; fi
done

PRED_OUT="$PROJECT_DIR/data/test_predictions/tta_on"
PRED_OUT_PP="$PROJECT_DIR/data/test_predictions/tta_on_pp"
VAL_FOR_PP="$MODEL_DIR_RESOLVED/val_for_pp"
mkdir -p "$PRED_OUT" "$PRED_OUT_PP"

echo "🔎 模型目录: $MODEL_DIR_RESOLVED"
echo "🔎 预测输入: $IMAGES_TS_RESOLVED"
echo "🔎 预测输出: $PRED_OUT (TTA启用)"
echo "🔎 预测后处理输出: $PRED_OUT_PP"

if [[ "${USE_VALIDATION_PREDS:-0}" == "1" ]]; then
  echo "🟡 使用训练时的 validation 预测作为后处理参考与可视化原始结果"
  VAL_FOR_PP="$FOLD_DIR/validation"
  if [[ ! -d "$VAL_FOR_PP" ]] || [[ -z "$(ls -A "$VAL_FOR_PP" 2>/dev/null)" ]]; then
    echo "❌ 未找到有效的 validation 预测: $VAL_FOR_PP"; exit 1
  fi
  # 使用部分 validation 预测作为可视化的原始预测，复制到 PRED_OUT
  echo "📥 拷贝部分 validation 预测到: $PRED_OUT"
  rm -f "$PRED_OUT"/* 2>/dev/null || true
  shopt -s nullglob
  cnt=0
  for f in "$VAL_FOR_PP"/*; do
    base="$(basename "$f")"; cp -f "$f" "$PRED_OUT/$base"; cnt=$((cnt+1));
    if [ "$cnt" -ge "$SAMPLE_COUNT" ]; then break; fi
  done
  echo "🧹 简单后处理（移除小连通域） -> $PRED_OUT_PP"
  python "$SCRIPT_DIR/simple_postprocess_masks.py" -i "$PRED_OUT" -o "$PRED_OUT_PP" --area_th 200 --closing 0 || {
    echo "❌ 简单后处理失败"; exit 1; }
else
  # 1) 生成用于确定后处理的预测（对 imagesTr 做带TTA预测），使用最佳权重以避免必须存在 checkpoint_final
  echo "🔁 生成用于确定后处理的预测 (样本 imagesTr, checkpoint_best.pth, 无多进程)..."
  mkdir -p "$VAL_FOR_PP"
  python "$SCRIPT_DIR/predict_no_mp.py" -i "$SAMPLE_TR_DIR" -o "$VAL_FOR_PP" -d "$DATASET_NAME_OR_ID" -c "$CONFIGURATION" -tr "$TRAINER" -p "$PLANS" -f all -chk checkpoint_best.pth || {
    echo "❌ 生成用于确定后处理的预测失败"; exit 1; }
fi

# 2) 自动确定后处理（基于上一步 imagesTr 的预测与 labelsTr）
if [[ "${USE_VALIDATION_PREDS:-0}" != "1" ]]; then
  echo "🧠 确定后处理 (postprocessing.pkl) ..."
  nnUNetv2_determine_postprocessing -i "$VAL_FOR_PP" -ref "$LABELS_TR_RESOLVED" -p "$PLANS_JSON" -d "$DATASET_JSON" -np "$NUM_PROCS" || {
    echo "❌ 确定后处理失败"; exit 1; }

  PP_PKL="$VAL_FOR_PP/postprocessing.pkl"
  if [[ ! -f "$PP_PKL" ]]; then
    echo "❌ 未找到 postprocessing.pkl: $PP_PKL"; exit 1
  fi
  echo "✅ 生成: $PP_PKL"
fi

if [[ "${USE_VALIDATION_PREDS:-0}" != "1" ]]; then
  # 3) 启用 TTA 的测试集/示例集预测（不使用 --disable_tta 即可开启镜像TTA）
  echo "🔁 运行预测 (样本, 无多进程)..."
  python "$SCRIPT_DIR/predict_no_mp.py" -i "$SAMPLE_TS_DIR" -o "$PRED_OUT" -d "$DATASET_NAME_OR_ID" -c "$CONFIGURATION" -tr "$TRAINER" -p "$PLANS" -f all -chk checkpoint_best.pth || {
    echo "❌ 预测失败"; exit 1; }
fi

# 4) 应用自动确定的后处理到上述预测结果
if [[ "${USE_VALIDATION_PREDS:-0}" != "1" ]]; then
  echo "🧹 应用后处理到预测结果..."
  nnUNetv2_apply_postprocessing -i "$PRED_OUT" -o "$PRED_OUT_PP" -pp_pkl_file "$PP_PKL" -np "$NUM_PROCS" -plans_json "$PLANS_JSON" -dataset_json "$DATASET_JSON" || {
    echo "❌ 应用后处理失败"; exit 1; }
fi

echo "✅ 完成: 带TTA预测 + 自动后处理"
echo "📁 预测输出: $PRED_OUT"
echo "📁 后处理输出: $PRED_OUT_PP"
