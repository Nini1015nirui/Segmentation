#!/bin/bash

# nnUNet preprocessing script for Dataset 803 (TA)
# Note: Run setup_training_environment.sh first to configure environment

echo "🔧 nnUNet数据预处理 - Dataset 803 (TA)"

# Check if environment is properly configured
if [ -z "$nnUNet_raw" ] || [ -z "$nnUNet_preprocessed" ] || [ -z "$nnUNet_results" ]; then
    echo "⚠️  环境变量未设置，请先运行: bash setup_training_environment.sh"
    exit 1
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "seg" ]; then
    echo "⚠️  请先激活conda环境: conda activate seg"
    exit 1
fi

echo "📍 Environment verification:"
echo "   nnUNet_raw: $nnUNet_raw"
echo "   nnUNet_preprocessed: $nnUNet_preprocessed"
echo "   nnUNet_results: $nnUNet_results"

echo "📊 Dataset verification:"
echo "   Training images: $(find $nnUNet_raw/Dataset803_TA/imagesTr -name "*.png" | wc -l)"
echo "   Training labels: $(find $nnUNet_raw/Dataset803_TA/labelsTr -name "*.png" | wc -l)"
echo "   Dataset.json exists: $([ -f $nnUNet_raw/Dataset803_TA/dataset.json ] && echo "Yes" || echo "No")"

echo ""
echo "🚀 开始nnUNet数据预处理..."
echo "   执行命令: nnUNetv2_plan_and_preprocess -d 803 --verify_dataset_integrity"
echo "   这将分析数据集并生成训练计划..."

# Run preprocessing with error handling
if nnUNetv2_plan_and_preprocess -d 803 --verify_dataset_integrity; then
    echo ""
    echo "✅ 预处理成功完成!"
    echo "📁 预处理数据位置: $nnUNet_preprocessed/Dataset803_TA/"
    echo "📋 现在可以开始训练: bash scripts/start_training.sh"
else
    echo ""
    echo "❌ 预处理失败，请检查数据集和环境配置"
    exit 1
fi
