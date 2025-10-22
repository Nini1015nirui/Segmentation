#!/usr/bin/env bash
set -eo pipefail

# Disable strict variable checking temporarily for conda activation
set +u

# Activate conda env
# Try to find and activate conda automatically
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)" || true
elif [ -f "/opt/miniconda3/bin/conda" ]; then
  eval "$(/opt/miniconda3/bin/conda shell.bash hook)" || true
fi
conda activate seg >/dev/null 2>&1

# Re-enable strict mode after conda activation
set -u

# Project-rooted nnUNet paths (use relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"

# Optimized memory settings for 8GB GPU
export nnUNet_n_proc_DA="4"  # Reduced from 6 to 4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"  # Reduced split size
export CUDA_LAUNCH_BLOCKING="1"  # Debug CUDA errors
export PYTORCH_CUDA_ALLOC_CONF="backend:native"  # Use native allocator

# Enable LCC (Largest Connected Component) postprocessing during validation
export NNUNET_USE_LCC_POSTPROC="1"  # Keep only largest connected region in binary segmentation

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo "🔧 完整全新训练配置 (每次从头开始):"
echo "   GPU Memory: 8GB RTX 4060 Laptop"
echo "   训练模式: 完整全新训练 (不使用检查点)"
echo "   Batch Size: 自动优化"
echo "   Memory Strategy: 保守分配策略"

# Show environment summary
python - << 'PY'
import torch, os
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
    print(f'Available Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0))/1024**3:.1f}GB')
print('nnUNet_n_proc_DA:', os.environ.get('nnUNet_n_proc_DA'))
print('PYTORCH_CUDA_ALLOC_CONF:', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
PY

# Stop any stale trainings and clear GPU memory
echo "🧹 Cleaning up GPU processes..."
pkill -f "nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet" || true
sleep 3

# Clear any GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Clean existing checkpoints for fresh start
CHECKPOINT_DIR="$nnUNet_results/Dataset803_TA/nnUNetTrainerLightMUNet__nnUNetPlans__2d/fold_all"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "🗑️ Clearing existing checkpoints for fresh training..."
    rm -rf "$CHECKPOINT_DIR"/*
    echo "   Removed all files from: $CHECKPOINT_DIR"
fi

# Launch fresh complete training (no --c flag)
LOG_OUT="/tmp/nnunet_fresh_training_$(date +%Y%m%d_%H%M%S).out"
echo "🚀 Starting COMPLETE fresh 2D segmentation training..."
echo "   每次都是全新的完整训练 - 不使用检查点续训"

# Launch training without continuation flag for complete fresh start
CUDA_VISIBLE_DEVICES=0 nohup nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet -device cuda >> "$LOG_OUT" 2>&1 &
PID=$!

echo "✅ 完整全新训练已启动 PID=$PID"
echo "📝 日志文件: $LOG_OUT"
echo "📊 实时监控: tail -f $LOG_OUT"

# Point to the training log
LOG_DIR="$nnUNet_results/Dataset803_TA/nnUNetTrainerLightMUNet__nnUNetPlans__2d/fold_all"
echo "🗂️ 训练目录: $LOG_DIR"

echo ""
echo "🎯 完整训练特点:"
echo "   ✅ 每次执行都是全新训练 (epoch 0开始)"
echo "   ✅ 自动清理所有现有检查点"
echo "   ✅ 获得完整的训练曲线数据"
echo "   ✅ 内存优化配置 (4进程, 64MB分割)"
echo "   ✅ 预期训练时间: ~35-43小时 (500 epochs)"
echo ""
echo "🛑 停止训练: kill $PID"
echo "🏁 训练完成检查: ls -la $LOG_DIR"