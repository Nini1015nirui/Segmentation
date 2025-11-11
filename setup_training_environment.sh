#!/bin/bash

# Complete environment setup for LightM-UNet training with TA dataset
echo "🔧 Setting up LightM-UNet training environment..."

# Set current project directory
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure nnUNet environment variables to use current project
export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"

# Initialize conda and ensure seg environment exists
if [ -z "${CONDA_EXE}" ]; then
  # Try to find conda automatically
  if command -v conda >/dev/null 2>&1; then
    export CONDA_EXE="$(command -v conda)"
  elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    export CONDA_EXE="$HOME/miniconda3/bin/conda"
  elif [ -f "/opt/miniconda3/bin/conda" ]; then
    export CONDA_EXE="/opt/miniconda3/bin/conda"
  else
    echo "⚠️ Conda not found. Please set CONDA_EXE environment variable."
    exit 1
  fi
fi
eval "$(${CONDA_EXE} shell.bash hook)"

# Create seg env from YAML if missing
if ! conda info --envs | awk '{print $1}' | grep -qx "seg"; then
  echo "📦 Creating conda env 'seg' from env/seg-environment.yml..."
  conda env create -n seg -f "$PROJECT_DIR/env/seg-environment.yml"
fi

# Activate seg environment
conda activate seg

# Ensure LightM-UNet package (nnUNetv2 entrypoints) is installed
if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "📦 Installing LightM-UNet package (editable) to provide nnUNetv2_* entrypoints..."
  pip install -e "$PROJECT_DIR/lightm-unet" --no-deps
fi

echo "📍 Environment Variables:"
echo "   nnUNet_raw: $nnUNet_raw"
echo "   nnUNet_preprocessed: $nnUNet_preprocessed"
echo "   nnUNet_results: $nnUNet_results"

echo "🐍 Python Environment:"
echo "   Python version: $(python --version)"
echo "   Working directory: $(pwd)"

echo "📦 Key packages verification:"
python -c "import torch; print('   PyTorch:', torch.__version__)"

# Ensure Mamba-SSM and causal-conv1d presence (install if missing)
python - <<'PY'
import sys
try:
    import importlib.metadata as md
except Exception:
    import importlib_metadata as md

def has(pkg):
    try:
        md.version(pkg)
        return True
    except Exception:
        return False

missing = []
if not has('mamba-ssm'):
    missing.append('mamba-ssm==2.2.5')
if not has('causal-conv1d'):
    missing.append('causal-conv1d>=1.2,<2')
if missing:
    import subprocess
    print('   Installing missing packages:', ', '.join(missing))
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=False)
else:
    print('   Mamba-SSM & causal-conv1d: OK')
PY

python - <<'PY'
try:
    import importlib.metadata as md
except Exception:
    import importlib_metadata as md
def v(n):
    try:
        return md.version(n)
    except Exception:
        return 'unknown'
print('   Mamba-SSM:', v('mamba-ssm'))
print('   causal-conv1d:', v('causal-conv1d'))
try:
    import nnunetv2
    print('   nnUNetv2: Available')
except Exception:
    print('   nnUNetv2: Error')
PY

echo "   nnUNetv2_train path: $(command -v nnUNetv2_train || echo 'not found')"

echo "📊 Dataset Status:"
echo "   Dataset location: $nnUNet_raw/Dataset803_TA"
echo "   Training images: $(ls $nnUNet_raw/Dataset803_TA/imagesTr/*.png | wc -l)"
echo "   Training labels: $(ls $nnUNet_raw/Dataset803_TA/labelsTr/*.png | wc -l)"

echo "✅ Environment ready for training!"

# Export the environment variables for the current session
echo "export nnUNet_raw=\"$nnUNet_raw\"" >> ~/.bashrc
echo "export nnUNet_preprocessed=\"$nnUNet_preprocessed\"" >> ~/.bashrc
echo "export nnUNet_results=\"$nnUNet_results\"" >> ~/.bashrc
echo "🔄 Environment variables added to ~/.bashrc"

echo ""
echo "🎯 Ready for next steps:"
echo "   1. ✅ Dataset converted (3605 images)"
echo "   2. ⏭️  Run: nnUNetv2_plan_and_preprocess -d 803 --verify_dataset_integrity"
echo "   3. ⏭️  Train: nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet"
