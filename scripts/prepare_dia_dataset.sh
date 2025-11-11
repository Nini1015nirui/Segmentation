#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_PY="/root/miniconda3/envs/seg/bin/python"
if [ ! -x "$RUN_PY" ]; then
  RUN_PY="python"
fi

echo "🔄 将 DIA 数据集转换为 nnUNet v2 格式 (Dataset804_DIA)"
cd "$PROJECT_DIR"
"$RUN_PY" ./convert_DIA_to_nnUNet.py --src DIA --out data/nnUNet_raw --dataset-id 804 --dataset-name DIA --train-split 1.0
echo "✅ 转换完成：data/nnUNet_raw/Dataset804_DIA"

