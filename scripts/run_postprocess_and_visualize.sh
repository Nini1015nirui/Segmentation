#!/usr/bin/env bash
set -eo pipefail

# ========================================
# LightM-UNet 后处理 + 可视化一键脚本
# ========================================
# 功能：
#   1. 运行 nnU-Net 官方后处理（TTA + 自动后处理）
#   2. 对比可视化原始预测 vs 后处理预测
#
# 使用方法：
#   bash scripts/run_postprocess_and_visualize.sh          # 默认参数
#   bash scripts/run_postprocess_and_visualize.sh 10      # 指定可视化10个样本
# ========================================

# 激活环境
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
fi
conda activate seg >/dev/null 2>&1 || true
set -u

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
export PATH="/root/miniconda3/envs/seg/bin:${PATH}"
export PYTHONPATH="$PROJECT_DIR/lightm-unet:${PYTHONPATH:-}"
# Tame OpenMP/BLAS threads to avoid SHM errors in constrained envs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# 参数
DATASET_ID="${1:-803}"
CONFIGURATION="${2:-2d}"
TRAINER="${3:-nnUNetTrainerLightMUNet}"
NUM_VIS_SAMPLES="${4:-10}"

echo "========================================"
echo "🎯 LightM-UNet 后处理 + 可视化流程"
echo "========================================"
echo "数据集ID: $DATASET_ID"
echo "配置: $CONFIGURATION"
echo "训练器: $TRAINER"
echo "可视化样本数: $NUM_VIS_SAMPLES"
echo "开始时间: $(date)"
echo ""

# ========================================
# Step 1: 运行 nnU-Net 官方后处理流程
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 1: 运行 nnU-Net 后处理流程"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "这将执行以下操作："
echo "  1. 对训练集做 TTA 预测（用于确定后处理参数）"
echo "  2. 自动确定最优后处理策略"
echo "  3. 对测试集做 TTA 预测"
echo "  4. 应用后处理到预测结果"
echo ""

bash "$SCRIPT_DIR/apply_tta_and_postprocessing.sh" "$DATASET_ID" "$CONFIGURATION" "$TRAINER" || {
    echo "❌ 后处理流程失败"
    exit 1
}

echo ""
echo "✅ 后处理流程完成"
echo ""

# ========================================
# Step 2: 可视化对比
# ========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🖼️  Step 2: 可视化后处理效果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 设置路径
PRED_RAW_DIR="$PROJECT_DIR/data/test_predictions/tta_on"
PRED_PP_DIR="$PROJECT_DIR/data/test_predictions/tta_on_pp"
LABEL_DIR="$PROJECT_DIR/data/nnUNet_raw/Dataset${DATASET_ID}_*/labelsTr"
IMAGE_DIR="$PROJECT_DIR/data/nnUNet_raw/Dataset${DATASET_ID}_*/imagesTr"
OUTPUT_DIR="$PROJECT_DIR/data/postprocess_visualization"

# 解析通配符路径
LABEL_DIR_RESOLVED=$(ls -d $LABEL_DIR 2>/dev/null | head -n1 || true)
IMAGE_DIR_RESOLVED=$(ls -d $IMAGE_DIR 2>/dev/null | head -n1 || true)

if [[ -z "$LABEL_DIR_RESOLVED" ]]; then
    echo "❌ 未找到标签目录: $LABEL_DIR"
    exit 1
fi

if [[ -z "$IMAGE_DIR_RESOLVED" ]]; then
    echo "❌ 未找到图像目录: $IMAGE_DIR"
    exit 1
fi

echo "原始预测: $PRED_RAW_DIR"
echo "后处理预测: $PRED_PP_DIR"
echo "标签目录: $LABEL_DIR_RESOLVED"
echo "图像目录: $IMAGE_DIR_RESOLVED"
echo "输出目录: $OUTPUT_DIR"
echo ""

python "$SCRIPT_DIR/visualize_with_postprocessing.py" \
    --pred_raw_dir "$PRED_RAW_DIR" \
    --pred_pp_dir "$PRED_PP_DIR" \
    --label_dir "$LABEL_DIR_RESOLVED" \
    --image_dir "$IMAGE_DIR_RESOLVED" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_VIS_SAMPLES" || {
    echo "❌ 可视化失败"
    exit 1
}

echo ""
echo "✅ 可视化完成"
echo ""

# ========================================
# 完成总结
# ========================================
echo "========================================"
echo "🎉 全部流程完成！"
echo "========================================"
echo "完成时间: $(date)"
echo ""
echo "📁 输出文件位置："
echo "  📊 后处理参数: $(dirname "$PRED_PP_DIR")/*/val_for_pp/postprocessing.pkl"
echo "  🔮 原始预测: $PRED_RAW_DIR"
echo "  🧹 后处理预测: $PRED_PP_DIR"
echo "  🖼️  可视化图片: $OUTPUT_DIR/visualizations"
echo "  📈 统计JSON: $OUTPUT_DIR/postprocess_summary.json"
echo ""
echo "🔍 快速查看："
echo "  # 查看可视化图片"
echo "  ls -lh $OUTPUT_DIR/visualizations/"
echo ""
echo "  # 查看统计结果"
echo "  cat $OUTPUT_DIR/postprocess_summary.json | grep -A10 'statistics'"
echo "========================================"
