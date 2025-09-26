#!/usr/bin/env bash
set -eo pipefail

# Recreate and verify the `seg` conda environment for LightM-UNet
# Pins to a stable combo: Python 3.10, torch 2.2.1+cu121, mamba-ssm 2.2.5, causal-conv1d >=1.2

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
ENV_NAME="seg"

echo "[1/6] Initializing conda shell..."
# Initialize conda
# Try to find conda automatically if not set
if [ -z "${CONDA_EXE}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
  elif [ -f "/opt/miniconda3/bin/conda" ]; then
    CONDA_EXE="/opt/miniconda3/bin/conda"
  else
    echo "⚠️ Conda not found. Please set CONDA_EXE environment variable."
    exit 1
  fi
fi
eval "$(${CONDA_EXE} shell.bash hook)"

echo "[2/6] Removing old env (if exists): ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y || true

echo "[3/6] Creating env from env/seg-environment.yml"
conda env create -n "${ENV_NAME}" -f "${PROJECT_DIR}/env/seg-environment.yml"

echo "[4/6] Activating env and installing project"
conda activate "${ENV_NAME}"
python -V
echo "Configuring CUDA toolchain to match PyTorch (12.1)"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
nvcc --version || true
echo "Installing build toolchain (gcc/g++, nvcc, CUDA headers)"
conda install -y -c conda-forge gxx_linux-64=12.* gcc_linux-64=12.* || true
conda install -y -c nvidia cuda-nvcc=12.1.* cuda-cudart-dev=12.1.* cuda-toolkit=12.1.* || true

echo "Ensuring torch stack pinned to conda versions"
conda install -y -c pytorch -c nvidia pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 numpy==1.26.4 scipy==1.11.*

echo "Installing LightM-UNet (editable, no deps)"
pip install -e "${PROJECT_DIR}/lightm-unet" --no-deps

echo "Rebuilding mamba-ssm from source against current torch"
pip uninstall -y mamba-ssm || true
export MAX_JOBS=${MAX_JOBS:-$(nproc || echo 4)}
# Broad arch list for common NVIDIA GPUs; adjust if needed
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
# Build from local clone to ensure proper include dirs and flags
git clone https://github.com/state-spaces/mamba.git -b v2.2.5 "$PROJECT_DIR/third_party/mamba" || true
python - <<'PY'
import os, re
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level from scripts/ to project root
p = os.path.join(os.environ.get('PROJECT_DIR', project_dir), 'third_party', 'mamba', 'setup.py')
with open(p,'r',encoding='utf-8') as f:
    s=f.read()
s=s.replace('include_dirs=[Path(this_dir) / "csrc" / "selective_scan"]',
            'include_dirs=[Path(this_dir)/"csrc"/"selective_scan", Path(str(os.environ.get("CUDA_HOME","")))/"targets"/"x86_64-linux"/"include"]')
s=s.replace('"-std=c++17",', '"-std=c++17","-allow-unsupported-compiler",')
with open(p,'w',encoding='utf-8') as f:
    f.write(s)
print('Patched', p)
PY
pip install -e "$PROJECT_DIR/third_party/mamba" --no-build-isolation

echo "[5/6] Sanity check: torch, mamba_ssm, causal_conv1d"
python - <<'PY'
import sys
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
try:
    import mamba_ssm
    print('mamba_ssm: OK')
except Exception as e:
    print('mamba_ssm: FAIL ->', e)
    sys.exit(1)
try:
    import causal_conv1d
    print('causal_conv1d: OK')
except Exception as e:
    print('causal_conv1d: FAIL ->', e)
    print('Attempting to build causal-conv1d from source (v1.2.2)...')
    import os, subprocess, sys, glob, shutil
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_project_dir = os.path.dirname(script_dir)  # Go up one level from scripts/ to project root
    project_dir = os.environ.get('PROJECT_DIR', default_project_dir)
    src = f"{project_dir}/third_party/causal-conv1d"
    if not os.path.isdir(src):
        subprocess.run(['git','clone','https://github.com/Dao-AILab/causal-conv1d.git', src], check=False)
    # Checkout a version compatible with torch 2.2.x
    subprocess.run(['git','-C',src,'checkout','-q','v1.2.2'], check=False)
    # Patch setup.py
    p = os.path.join(src,'setup.py')
    s = open(p,'r',encoding='utf-8').read()
    s = s.replace('include_dirs=[Path(this_dir) / "csrc" / "causal_conv1d"]',
                  'include_dirs=[Path(this_dir)/"csrc"/"causal_conv1d", Path(os.environ.get("CUDA_HOME",""))/"targets"/"x86_64-linux"/"include"]')
    if 'allow-unsupported-compiler' not in s:
        s = s.replace('"-O3",', '"-O3","-allow-unsupported-compiler",')
    open(p,'w',encoding='utf-8').write(s)
    # Build extension
    try:
        subprocess.run([sys.executable,'setup.py','build_ext','-v'], check=True, cwd=src)
    except Exception as be:
        print('Build failed:', be)
    # Try pip editable install without build isolation (may still trigger PEP517)
    ok = subprocess.run(['pip','install','-e',src,'--no-build-isolation'], cwd=src).returncode == 0
    if not ok:
        # Manual staging fallback: copy python pkg + built .so into site-packages
        print('Editable install failed; manually staging package...')
        site = os.path.join(sys.prefix,'lib','python3.10','site-packages')
        for name in ('__editable__.causal_conv1d-1.5.2.pth','__editable___causal_conv1d_1_5_2_finder.py',
                     'causal_conv1d-1.5.2.dist-info','causal_conv1d-1.2.2.dist-info','causal_conv1d'):
            pth = os.path.join(site,name)
            if os.path.isfile(pth):
                os.remove(pth)
            elif os.path.isdir(pth):
                shutil.rmtree(pth, ignore_errors=True)
        shutil.copytree(os.path.join(src,'causal_conv1d'), os.path.join(site,'causal_conv1d'), dirs_exist_ok=True)
        so_candidates = glob.glob(os.path.join(src,'build','lib.*','causal_conv1d_cuda.*.so'))
        if so_candidates:
            shutil.copy2(so_candidates[0], os.path.join(site,'causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so'))
        else:
            print('Warning: built .so not found; import may fail')
    import importlib
    importlib.invalidate_caches()
    import causal_conv1d as cc
    print('causal_conv1d: OK after source build')
PY

echo "[6/6] Exporting nnUNet environment variables for this session"
export nnUNet_raw="${PROJECT_DIR}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/data/nnUNet_results"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

cat <<'MSG'

Done. Next steps:
- Preprocess: nnUNetv2_plan_and_preprocess -d 803 --verify_dataset_integrity
- Train:      nnUNetv2_train 803 2d all -tr nnUNetTrainerLightMUNet

Tip: To persist env vars, append to ~/.bashrc:
  export nnUNet_raw="$PROJECT_DIR/data/nnUNet_raw"
  export nnUNet_preprocessed="$PROJECT_DIR/data/nnUNet_preprocessed"
  export nnUNet_results="$PROJECT_DIR/data/nnUNet_results"
MSG
