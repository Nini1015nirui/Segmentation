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
export nnUNet_n_proc_DA="4"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export CUDA_LAUNCH_BLOCKING="1"
export PYTORCH_CUDA_ALLOC_CONF="backend:native"

# Check if checkpoint exists
CHECKPOINT_DIR="$nnUNet_results/Dataset803_TA/nnUNetTrainerLightMUNet__nnUNetPlans__2d/fold_all"
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
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo "🔄 继续训练配置:"
echo "   GPU Memory: 8GB RTX 4060 Laptop"
echo "   训练模式: 继续训练 (使用现有检查点)"
echo "   检查点目录: $CHECKPOINT_DIR"
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

# Check latest checkpoint info
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo ""
    echo "📊 最新检查点信息:"
    python - << PY
import torch
try:
    ckpt = torch.load("$CHECKPOINT_DIR/checkpoint_latest.pth", map_location='cpu')
    if 'current_epoch' in ckpt:
        print(f"   上次训练轮次: Epoch {ckpt['current_epoch']}")
    if 'ema_fg_dice' in ckpt['network_weights']:
        print(f"   EMA Dice: {ckpt['network_weights']['ema_fg_dice']:.4f}")
except Exception as e:
    print(f"   无法读取检查点信息: {e}")
PY
fi

# Stop any stale trainings and clear GPU memory
echo ""
echo "🧹 清理 GPU 进程..."
pkill -f "nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet" || true
sleep 3

# Clear any GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Launch continuation training with -c flag
LOG_OUT="/tmp/nnunet_resume_training_$(date +%Y%m%d_%H%M%S).out"
echo "🚀 继续 2D 分割训练..."
echo "   从上次中断处恢复训练"

# Launch training WITH continuation flag (--c)
CUDA_VISIBLE_DEVICES=0 nohup nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet --c -device cuda >> "$LOG_OUT" 2>&1 &
PID=$!

echo ""
echo "✅ 训练已恢复 PID=$PID"
echo "📝 临时日志: $LOG_OUT"
echo "📊 实时监控: tail -f $LOG_OUT"
echo "📂 训练日志: $CHECKPOINT_DIR/training_log_*.txt"

echo ""
echo "🎯 继续训练特点:"
echo "   ✅ 从上次中断的 epoch 继续"
echo "   ✅ 保留所有训练历史和检查点"
echo "   ✅ 自动加载最佳模型权重"
echo "   ✅ 内存优化配置 (4进程, 64MB分割)"
echo ""
echo "🛑 停止训练: kill $PID"
echo "🏁 训练状态检查: tail -20 $CHECKPOINT_DIR/training_log_*.txt"