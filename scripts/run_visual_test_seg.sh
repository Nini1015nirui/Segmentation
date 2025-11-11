#!/usr/bin/env bash
set -eo pipefail

# Run segmentation on 可视化测试 images using trained DIA model in seg env,
# then simple postprocess and create overlays.

# Usage:
#   bash scripts/run_visual_test_seg.sh                      # defaults
#   bash scripts/run_visual_test_seg.sh 可视化测试/Images    # custom images dir

IMAGES_DIR_INPUT="${1:-可视化测试/Images}"
MASKS_DIR_INPUT="${2:-可视化测试/Masks}"
DATASET_ID="${DATASET_ID:-804}"
CONFIGURATION="${CONFIGURATION:-2d}"
TRAINER="${TRAINER:-nnUNetTrainerLightMUNet}"
PLANS_IN="${PLANS:-}"  # empty -> auto
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
AREA_TH="${AREA_TH:-200}"
PP_KEEP_LCC="${PP_KEEP_LCC:-1}"
PP_LCC_PRE_ERODE="${PP_LCC_PRE_ERODE:-3}"

# Activate seg env if available
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
fi
conda activate seg >/dev/null 2>&1 || true
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export nnUNet_raw="${nnUNet_raw:-$PROJECT_DIR/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_DIR/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_DIR/data/nnUNet_results}"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
# Tame OpenMP/BLAS threads to avoid SHM issues in constrained envs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# Auto-pick plans (prefer mem4 if corresponding model/checkpoint exists)
if [[ -z "$PLANS_IN" ]]; then
  CAND_MEM4="$nnUNet_results/Dataset${DATASET_ID}_*/${TRAINER}__nnUNetPlans_mem4__${CONFIGURATION}/fold_all/checkpoint_best.pth"
  CAND_BASE="$nnUNet_results/Dataset${DATASET_ID}_*/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_all/checkpoint_best.pth"
  if ls $CAND_MEM4 >/dev/null 2>&1; then
    PLANS_IN="nnUNetPlans_mem4"
  elif ls $CAND_BASE >/dev/null 2>&1; then
    PLANS_IN="nnUNetPlans"
  elif ls "$nnUNet_preprocessed"/Dataset${DATASET_ID}_*/nnUNetPlans_mem4.json >/dev/null 2>&1; then
    PLANS_IN="nnUNetPlans_mem4"
  else
    PLANS_IN="nnUNetPlans"
  fi
fi

IMAGES_DIR="$IMAGES_DIR_INPUT"
MASKS_DIR="$MASKS_DIR_INPUT"

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "❌ Images dir not found: $IMAGES_DIR"; exit 1
fi

OUT_BASE="$PROJECT_DIR/data/visual_test_predictions"
RAW_DIR="$OUT_BASE/raw"
PP_DIR="$OUT_BASE/pp"
VIZ_DIR="$OUT_BASE/viz"
mkdir -p "$RAW_DIR" "$PP_DIR" "$VIZ_DIR"

echo "========================================"
echo "🧪 可视化测试分割 (DIA model)"
echo "========================================"
echo "Images: $IMAGES_DIR"
echo "Masks (optional GT): $MASKS_DIR"
echo "Out (raw): $RAW_DIR"
echo "Out (pp):  $PP_DIR"
echo "Out (viz): $VIZ_DIR"
echo "Dataset ID: $DATASET_ID  Config: $CONFIGURATION  Trainer: $TRAINER  Plans: $PLANS_IN"
echo "Checkpoint: $CHECKPOINT_NAME"

echo "\n🔁 Predicting with TTA ON (single-process)..."
python "$SCRIPT_DIR/predict_no_mp.py" \
  -i "$IMAGES_DIR" \
  -o "$RAW_DIR" \
  -d "Dataset${DATASET_ID}_DIA" \
  -c "$CONFIGURATION" \
  -tr "$TRAINER" \
  -p "$PLANS_IN" \
  -f all \
  -chk "$CHECKPOINT_NAME" \
  --pattern "*.png"

if [[ "$PP_KEEP_LCC" == "1" ]]; then
  echo "\n🧹 Simple postprocess -> keep largest connected component (LCC) (pre_erode=${PP_LCC_PRE_ERODE})"
  python "$SCRIPT_DIR/simple_postprocess_masks.py" -i "$RAW_DIR" -o "$PP_DIR" --keep_lcc 1 --closing 0 --lcc_pre_erode "$PP_LCC_PRE_ERODE" --lcc_post_dilate "$PP_LCC_PRE_ERODE"
else
  echo "\n🧹 Simple postprocess -> remove small islands (area<$AREA_TH)"
  python "$SCRIPT_DIR/simple_postprocess_masks.py" -i "$RAW_DIR" -o "$PP_DIR" --area_th "$AREA_TH" --closing 0 --keep_lcc 0
fi

echo "\n🖼️  Create overlays (Original | PP vs GT)"
python "$SCRIPT_DIR/vis_pp_overlay.py" \
  --images_dir "$IMAGES_DIR" \
  --pp_pred_dir "$PP_DIR" \
  --gt_dir "$MASKS_DIR" \
  --out_dir "$VIZ_DIR" \
  --pp_vs_gt 1

echo "\n✅ Done"
echo "Raw predictions: $RAW_DIR"
echo "Post-processed:  $PP_DIR"
echo "Overlays:        $VIZ_DIR"
