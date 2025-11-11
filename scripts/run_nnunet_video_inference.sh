#!/usr/bin/env bash
set -euo pipefail

# nnUNet inference on a DICOM cine video using the trained DIA model (seg env)

# Usage:
#   bash scripts/run_nnunet_video_inference.sh '测试vedio' ceshi_vedio
#   # or custom fps
#   FPS=25 bash scripts/run_nnunet_video_inference.sh input.dcm run_name

VIDEO_IN="${1:?Input DICOM required}"
RUN_NAME="${2:-video_run}"
FPS="${FPS:-25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate seg env (if available)
if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh || true
  conda activate seg >/dev/null 2>&1 || true
fi

# nnUNet paths
export nnUNet_raw="${nnUNet_raw:-$PROJECT_DIR/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_DIR/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_DIR/data/nnUNet_results}"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
# Tame threads
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

WORKDIR="$PROJECT_DIR/data/video_output/$RUN_NAME"
FRAMES="$WORKDIR/frames"
RAW_DIR="$WORKDIR/nnunet_raw"
PP_DIR="$WORKDIR/nnunet_pp"
OVERLAY_FRAMES="$WORKDIR/overlays_nnunet"
mkdir -p "$FRAMES" "$RAW_DIR" "$PP_DIR" "$OVERLAY_FRAMES"

echo "[1/5] Extract frames via ffmpeg -> $FRAMES"
ffmpeg -y -i "$VIDEO_IN" "$FRAMES/frame_%06d.png"

echo "[2/5] Pick plans according to existing checkpoints"
PLANS="nnUNetPlans"
if ls "$nnUNet_results"/Dataset804_DIA/nnUNetTrainerLightMUNet__nnUNetPlans_mem4__2d/fold_all/checkpoint_best.pth >/dev/null 2>&1; then
  PLANS="nnUNetPlans_mem4"
fi
echo "Plans: $PLANS; Trainer: nnUNetTrainerLightMUNet (2d)"

echo "[3/5] Predict masks with your trained DIA model"
python "$SCRIPT_DIR/predict_no_mp.py" \
  -i "$FRAMES" \
  -o "$RAW_DIR" \
  -d Dataset804_DIA \
  -c 2d \
  -tr nnUNetTrainerLightMUNet \
  -p "$PLANS" \
  -f all \
  -chk checkpoint_best.pth \
  --pattern "*.png"

echo "[4/5] Simple postprocess: keep LCC"
python "$SCRIPT_DIR/simple_postprocess_masks.py" -i "$RAW_DIR" -o "$PP_DIR" --keep_lcc 1 --lcc_pre_erode 3 --lcc_post_dilate 3

echo "[5/5] Overlay + assemble MP4"
python "$SCRIPT_DIR/overlay_masks_to_video.py" \
  --frames_dir "$FRAMES" \
  --masks_dir "$PP_DIR" \
  --out_frames_dir "$OVERLAY_FRAMES" \
  --out_mp4 "$WORKDIR/overlays_nnunet.mp4" \
  --fps "$FPS"

ffmpeg -y -i "$VIDEO_IN" -c:v libx264 -crf 18 -pix_fmt yuv420p "$WORKDIR/original.mp4"

echo "\n✅ Done"
echo "Frames:          $FRAMES"
echo "Raw masks:       $RAW_DIR"
echo "Postprocessed:   $PP_DIR"
echo "Overlay frames:  $OVERLAY_FRAMES"
echo "Original MP4:    $WORKDIR/original.mp4"
echo "Overlay MP4:     $WORKDIR/overlays_nnunet.mp4"

