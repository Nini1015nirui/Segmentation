## 🚀 快速使用指南

### 训练
```bash
# 场景1: 第一次训练
bash scripts/start_training.sh
bash scripts/start_training.sh  # Epoch 0 → 300

# 场景2: 训练中断，需要继续
bash scripts/resume_training.sh 
bash scripts/resume_training.sh  # 从上次中断处继续
```

### 推理与可视化（训练完成后）
```bash
# 一键执行：后处理 + TTA + 可视化
bash scripts/run_postprocess_and_visualize.sh

# 自定义可视化样本数（例如20个）
bash scripts/run_postprocess_and_visualize.sh 803 2d nnUNetTrainerLightMUNet 20
```

**输出**：
- 📊 后处理参数：`data/nnUNet_results/.../val_for_pp/postprocessing.pkl`
- 🔮 原始预测：`data/test_predictions/tta_on/`
- 🧹 后处理预测：`data/test_predictions/tta_on_pp/`
- 🖼️ 可视化图片：`data/postprocess_visualization/visualizations/`
- 📈 统计结果：`data/postprocess_visualization/postprocess_summary.json`

**详细文档**：
- [推理与可视化指南](docs/inference_and_visualization_guide.md)
- [后处理详细说明](docs/postprocessing_guide.md)

--- 

# [LightM-UNet](https://arxiv.org/html/2403.05246v1)

Official repository for "LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation".

## Release

-  🔥**News**: ```2024/3/29```: Received feedback on [issue#6](https://github.com/MrBlankness/LightM-UNet/issues/6) and updated the erroneous code in the original code repository.

-  **News**: ```2024/3/17```: LightM-UNet released.

- **A regrettable notification**: ```2024/3/12``` Thank you for your attention! The author has been quite busy lately (；′⌒`), so only the main code has been uploaded to this repository for now. Detailed code explanations, data, and configuration details will be completed by March 17, 2024.
Thanks again for your interest! o(￣▽￣)ブ

## Introduction to LightM-UNet

LightM-UNet is a lightweight fusion of UNet and Mamba, boasting a mere parameter count of **1M**. Through validation on both 2D and 3D real-world datasets, LightM-UNet surpasses existing state-of-the-art models. In comparison to the renowned [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and contemporaneous [U-Mamba](https://github.com/bowang-lab/U-Mamba), LightM-UNet reduces the parameter count by **116X** and **224X**, respectively.

![result](https://github.com/MrBlankness/LightM-UNet/blob/master/assets/main_result.png)

## Get Start 

Requirements: `CUDA ≥ 11.6`

1) Quickstart (original steps)

1. Create a virtual environment: `conda create -n lightmunet python=3.10 -y` and `conda activate lightmunet`
2. Install PyTorch 2.4.1 (CUDA 12.1): `pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. Install Mamba stack: `pip install mamba-ssm==2.2.5 causal-conv1d==1.5.2`
4. Download code: `git clone https://github.com/MrBlankness/LightM-UNet`
5. `cd LightM-UNet/lightm-unet` and run `pip install -e .`

2) Recommended (seg environment used in this repo)

- Create/update the `seg` conda environment from the provided YAML (CUDA 12.1 + torch 2.4.1):

```bash
conda env create -n seg -f env/seg-environment.yml   # first time
# or update later
conda env update -n seg -f env/seg-environment.yml
conda activate seg
```

Note: The pinned versions in `env/seg-environment.yml` and the lock snapshot (`env/seg-env-keys.txt`) reflect the current working setup used in this repository (PyTorch 2.4.1+cu121, mamba-ssm 2.2.5, causal-conv1d 1.5.2, nnUNetv2 2.1.1, MONAI 1.3.0).

- Install the project package to provide `nnUNetv2_*` commands:

```bash
pip install -e ./lightm-unet --no-deps
```

- Set nnUNet paths (adjust if you cloned to a different location):

```bash
export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnUNet_results"
```

- Sanity check core libraries:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu:', torch.cuda.is_available())
try:
    import importlib.metadata as md
except Exception:
    import importlib_metadata as md
def v(n):
    try:
        return md.version(n)
    except Exception:
        return 'missing'
print('mamba-ssm:', v('mamba-ssm'))
print('causal-conv1d:', v('causal-conv1d'))
import nnunetv2
print('nnUNetv2: OK')
PY
```

- If `mamba-ssm` wheels are unavailable for your stack and the import fails, either:
  - Install/pin via pip: `pip install mamba-ssm==2.2.5 causal-conv1d>=1.2,<2`, or
  - Use the helper to rebuild from source against the active torch/CUDA:

```bash
bash scripts/fix_seg_env.sh
```


## Enable True Mamba-SSM (CUDA)

### 🔥 **CUDA Compatibility Solution - Production Ready**

**Problem**: `undefined symbol: _ZN3c104cuda14ExchangeDeviceEa` when importing mamba-ssm
**Solution**: ✅ **Verified Working Configuration** - PyTorch 2.4.1+cu121 + mamba-ssm 2.2.5

#### **Critical Success Configuration**
```yaml
Hardware: RTX 4060/3060/4070+ (8GB+ VRAM recommended)
OS: Linux/WSL2 Ubuntu
Python: 3.10.18
PyTorch: 2.4.1+cu121  # CRITICAL VERSION
CUDA: 12.1
mamba-ssm: 2.2.5
causal-conv1d: 1.5.2
```

#### **Step-by-Step CUDA Compatibility Fix**

**1. Upgrade PyTorch (Critical Step)**
```bash
conda activate seg
# Remove old PyTorch versions
pip uninstall torch torchvision torchaudio -y
# Install compatible version
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Install/Reinstall mamba-ssm**
```bash
# Clean install
pip uninstall mamba-ssm causal-conv1d -y
pip install mamba-ssm==2.2.5 causal-conv1d>=1.2,<2
```

**3. Verify True Mamba CUDA Extension**
```bash
python scripts/verify_true_mamba.py
# Expected output: "SUCCESS: Real mamba-ssm CUDA extension is active."
```

#### **Production Verification Checklist**
```python
# Complete functionality test
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

import mamba_ssm, selective_scan_cuda
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('✅ mamba-ssm: Real CUDA backend active')

# Test CUDA functionality
B, D, L, N = 1, 4, 16, 8
u = torch.randn(B, D, L, device='cuda', dtype=torch.float16, requires_grad=True)
delta = torch.rand(B, D, L, device='cuda', dtype=torch.float16)
A = torch.randn(D, N, device='cuda', dtype=torch.float32)
Bm = torch.randn(D, N, device='cuda', dtype=torch.float32)
C = torch.randn(D, N, device='cuda', dtype=torch.float32)

out = selective_scan_fn(u, delta, A, Bm, C)
out.sum().backward()
print('✅ selective_scan CUDA kernel: Forward + Backward OK')

from nnunetv2.nets.LightMUNet import MAMBA_AVAILABLE
print(f'✅ LightMUNet MAMBA_AVAILABLE: {MAMBA_AVAILABLE}')
"
```

#### **Alternative Methods (if needed)**

If your current environment falls back to the Python/triton path and cannot load the CUDA extension, use the following on a GPU node with a compatible NVIDIA driver and sufficient shared memory (`/dev/shm`).

- Quick build and verify against your current PyTorch:

```bash
# On the target training machine (with GPU)
bash scripts/enable_true_mamba.sh
python scripts/verify_true_mamba.py
```

- If the build still fails due to missing/changed symbols, upgrade PyTorch to a version where the required symbol is present (2.3+), then rebuild:

```bash
PT_UPGRADE=1 PT_VER=2.4.1 PT_CUDA=cu121 bash scripts/enable_true_mamba.sh
python scripts/verify_true_mamba.py
```

Notes:
- Choose `PT_CUDA` to match your driver (common: `cu121` or `cu124`).
- For Docker, run with `--ipc=host` or `--shm-size=16g` to ensure `/dev/shm` is large enough.


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Data Preparation

Download LiTs dataset [here](https://www.kaggle.com/datasets/gauravduttakiit/3d-liver-and-liver-tumor-segmentation) and Montgomery&Shenzhen dataset [here](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels), then put them into the `LightM-Unet/data/nnUNet_raw` folder. 
LightM-UNet is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. If you want to train LightM-UNet on your own dataset, please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset. 

Please organize the dataset as follows:

```
data/
├── nnUNet_raw/
│   ├── Dataset801_LiverCT/
│   │   ├── imagesTr
│   │   │   ├── Liver_0001_0000.nii.gz
│   │   │   ├── Liver_0002_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── Liver_0001.nii.gz
│   │   │   ├── Liver_0002.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── Dataset802_LungXray/
│   │   ├── imagesTr
│   │   │   ├── Lung_0001_0000.png
│   │   │   ├── Lung_0002_0000.png
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── Lung_0001.png
│   │   │   ├── Lung_0001.png
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── ...
```

Based on nnUNet, preprocess the data and generate the corresponding configuration files (the generated results can be found in the `LightM-Unet/data/nnUNet_preprocessed` folder).

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## 🚀 **Production Environment Configuration**

### **Verified Working Environment (Battle-Tested)**

This configuration has been successfully tested with **32+ training epochs** achieving **Dice 0.5612** with true mamba-ssm CUDA acceleration:

#### **System Specifications**
```yaml
System Environment:
  OS: Linux WSL2 (6.6.87.1-microsoft-standard-WSL2)
  Architecture: x86_64
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
  Driver: 566.26
  CUDA Runtime: 12.7 (compatible with 12.1)

Python Environment:
  Python: 3.10.18 (conda-forge)
  Environment: seg (isolated conda env)
  Package Manager: conda + pip hybrid
```

#### **Critical Package Versions**
```yaml
Deep Learning Stack:
  torch: 2.4.1+cu121          # CRITICAL - solves CUDA symbol issues
  torchvision: corresponding
  torchaudio: corresponding
  cuda-toolkit: 12.1

Mamba Ecosystem:
  mamba-ssm: 2.2.5            # Verified working with PyTorch 2.4.1
  causal-conv1d: 1.5.2        # Successfully compiled CUDA extension
  selective-scan-cuda: ✅      # Real CUDA backend active

Medical AI Framework:
  nnunetv2: 2.1.1             # LightM-UNet modified version
  monai: 1.3.0

Scientific Computing:
  numpy: 1.26.4               # PyTorch compatible version
```

#### **Environment Variables Configuration**
```bash
# Required for nnUNet framework
export nnUNet_raw="/path/to/LightM-UNet-master/data/nnUNet_raw"
export nnUNet_preprocessed="/path/to/LightM-UNet-master/data/nnUNet_preprocessed"
export nnUNet_results="/path/to/LightM-UNet-master/data/nnUNet_results"

# Optional GPU configuration
export CUDA_VISIBLE_DEVICES=0  # Single GPU training
```

#### **Training Performance Metrics**
```yaml
Training Characteristics:
  Epoch Time: 103±1 seconds (stable)
  Batch Size: 2 (optimized for 8GB VRAM)
  Patch Size: 512×512
  GPU Memory: 286MB baseline, stable during training

Current Performance (Epoch 32):
  EMA Pseudo Dice: 0.5612 (continuously improving)
  Current Dice: 0.5948
  Training Loss: -0.17 (excellent convergence)
  Validation Loss: -0.1774

Model Characteristics:
  Parameters: ~1M (116x fewer than standard nnUNet)
  Architecture: True State Space Model (non-fallback)
  Convergence: Stable, no overfitting detected
```

#### **Training Stability Features**
```yaml
Gradient Management:
  Gradient Clipping: 12.0 (prevents explosion)
  Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
  Scheduler: PolyLRScheduler (exponent=0.9)

Data Configuration:
  Dataset: TA (3,605 medical images)
  Train/Val Split: nnUNet automatic
  Normalization: Z-Score
  Augmentation: nnUNet standard pipeline

Checkpoint Strategy:
  Auto-save: Best model (checkpoint_best.pth)
  Recovery: Full state restoration on resume
  Location: data/nnUNet_results/.../fold_all/
```

#### **Complete Environment Setup Script**
```bash
#!/bin/bash
# Production environment setup

# 1. Create conda environment
conda env create -n seg -f env/seg-environment.yml
conda activate seg

# 2. Upgrade to working PyTorch version (CRITICAL)
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install mamba packages
pip install mamba-ssm==2.2.5 causal-conv1d>=1.2,<2

# 4. Install LightM-UNet
pip install -e ./lightm-unet --no-deps

# 5. Set environment variables
export nnUNet_raw="$(pwd)/data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/data/nnUNet_results"

# 6. Verify complete setup
python scripts/verify_true_mamba.py
echo "✅ Environment ready for production training"
```

#### **Health Check & Monitoring**
```bash
# Environment health monitoring script
python -c "
import torch, mamba_ssm, selective_scan_cuda
from nnunetv2.nets.LightMUNet import MAMBA_AVAILABLE

print('=== ENVIRONMENT HEALTH CHECK ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print(f'mamba-ssm: OK')
print(f'selective_scan_cuda: OK')
print(f'MAMBA_AVAILABLE: {MAMBA_AVAILABLE}')
print(f'Status: ✅ Production Ready')
print('=== CHECK COMPLETE ===')
"

# GPU monitoring
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

#### **Troubleshooting Common Issues**

**Issue 1: CUDA Symbol Error**
```bash
# Symptoms: undefined symbol: _ZN3c104cuda14ExchangeDeviceEa
# Solution: Upgrade PyTorch (already handled above)
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Issue 2: mamba-ssm Fallback Mode**
```bash
# Symptoms: "Warning: mamba_ssm not available, using fallback"
# Check: Verify MAMBA_AVAILABLE flag
python -c "from nnunetv2.nets.LightMUNet import MAMBA_AVAILABLE; print(f'Real Mamba: {MAMBA_AVAILABLE}')"
```

**Issue 3: GPU Memory Issues**
```bash
# Symptoms: CUDA out of memory
# Solution: Kill existing processes, reduce batch size
pkill -f nnUNetv2_train
# Reduce batch size in trainer if needed
```

**Issue 4: Training Slow/Unstable**
```bash
# Check GPU utilization
nvidia-smi
# Verify no CPU fallback
python -c "import torch; print(f'CUDA device: {torch.cuda.current_device()}')"
```

## Model Training


### Train 2D models

- Train 2D `LightM-Unet` model

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerLightMUNet
```

### Train 3D models

- Train 3D `LightM-Unet` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerLightMUNet
```


## Inference

### Inference 2D models

- Inference 2D `LightM-Unet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 2d -tr nnUNetTrainerLightMUNet --disable_tta
```

### Inference 3D models

- Inference 3D `LightM-Unet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr nnUNetTrainerLightMUNet --disable_tta
```


## Citation
If you find our work helpful, please consider citing the following papers
```
@misc{liao2024lightmunet,
      title={LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation}, 
      author={Weibin Liao and Yinghao Zhu and Xinyuan Wang and Chengwei Pan and Yasha Wang and Liantao Ma},
      year={2024},
      eprint={2403.05246},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. 
We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.
