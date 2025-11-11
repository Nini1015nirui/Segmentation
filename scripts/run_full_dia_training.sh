#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== [1/3] Convert DIA → nnUNet ==="
bash "$SCRIPT_DIR/prepare_dia_dataset.sh"

echo "=== [2/3] Plan & Preprocess ==="
# For environments with limited shared memory, keep OMP threads minimal
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
bash "$SCRIPT_DIR/preprocess_dia.sh"

echo "=== [3/3] Start Full Training ==="
bash "$SCRIPT_DIR/start_training_dia.sh"

echo "✅ Pipeline submitted. Use tail -f /tmp/nnunet_DIA_training_*.out to monitor."

