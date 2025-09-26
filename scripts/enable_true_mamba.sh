#!/usr/bin/env bash
set -euo pipefail

# Enable the real CUDA mamba-ssm extension (no fallback) on a GPU node.
# Strategy:
#   1) Try building mamba-ssm 2.2.5 from source against the current PyTorch/CUDA.
#   2) If that still fails to import CUDA ops, optionally upgrade PyTorch (>=2.3) and retry.
#
# Usage:
#   bash scripts/enable_true_mamba.sh                # build from source, verify
#   PT_UPGRADE=1 bash scripts/enable_true_mamba.sh   # also upgrade PyTorch (cu121 by default)
#   PT_VER=2.4.1 PT_CUDA=cu121 PT_UPGRADE=1 bash scripts/enable_true_mamba.sh
#
# Notes:
#   - Run on a GPU machine with matching NVIDIA driver for the chosen CUDA runtime.
#   - Ensure sufficient shared memory (/dev/shm, e.g. --ipc=host or --shm-size=16g for Docker).

PT_UPGRADE=${PT_UPGRADE:-0}           # 0 or 1 – whether to upgrade Torch if build fails
PT_VER=${PT_VER:-2.4.1}               # Torch version to install when upgrading
PT_CUDA=${PT_CUDA:-cu121}             # One of: cu121, cu124 (depends on driver)
MAX_JOBS=${MAX_JOBS:-$(nproc || echo 4)}

banner() { echo -e "\n==== $* ===="; }

python - <<'PY'
import os, sys, subprocess, shutil
print('Python:', sys.version)
try:
    import torch
    print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device count:', torch.cuda.device_count())
        from subprocess import run, PIPE
        r = run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], stdout=PIPE, stderr=PIPE, text=True)
        if r.returncode == 0:
            caps = ';'.join(sorted(set(x.strip() for x in r.stdout.splitlines() if x.strip())))
            print('Compute capability:', caps)
except Exception as e:
    print('Torch import failed:', e)
PY

banner "Check /dev/shm (shared memory)"
df -h /dev/shm || true
mount | grep /dev/shm || true

banner "Install build toolchain (pip/setuptools/wheel/ninja/cmake)"
python -m pip install -U pip setuptools wheel packaging ninja cmake --extra-index-url https://download.pytorch.org/whl/${PT_CUDA} || true

# Derive arch list from the live GPU when possible (e.g., 8.6)
ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk '{gsub(",","."); print $1}' | sort -u | paste -sd ';' -)
if [[ -z "${ARCH_LIST}" ]]; then
  # Fallback to common recent archs
  ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
fi
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export MAX_JOBS
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

banner "Attempt 1: build mamba-ssm 2.2.5 from source"
python -m pip uninstall -y mamba-ssm || true
# Build in-place against the current torch without PEP517 isolation
python -m pip install --no-binary=:all: --no-build-isolation --no-cache-dir \
  "mamba-ssm==2.2.5" "causal-conv1d>=1.2,<2" || true

banner "Verify CUDA extension import"
set +e
python - <<'PY'
import sys
ok = True
try:
    import mamba_ssm
    from mamba_ssm import ops
    import selective_scan_cuda  # compiled CUDA extension
    print('mamba-ssm import: OK, ops/selective_scan_cuda present')
except Exception as e:
    ok = False
    print('mamba-ssm CUDA import failed:', e)
sys.exit(0 if ok else 42)
PY
CODE=$?
set -e

if [[ ${CODE} -ne 0 && "${PT_UPGRADE}" == "1" ]]; then
  banner "Attempt 2: upgrade PyTorch to >=2.3 (${PT_VER} ${PT_CUDA}), then reinstall"
  # Remove current torch stack to avoid ABI mismatches
  python -m pip uninstall -y torch torchvision torchaudio || true
  python -m pip install --index-url https://download.pytorch.org/whl/${PT_CUDA} "torch==${PT_VER}" torchvision torchaudio

  # Rebuild/reinstall mamba-ssm against the new torch
  python -m pip uninstall -y mamba-ssm || true
  python -m pip install --no-binary=:all: --no-build-isolation --no-cache-dir \
    "mamba-ssm==2.2.5" "causal-conv1d>=1.2,<2"

  banner "Re-verify CUDA extension import"
  python - <<'PY'
import sys
ok = True
try:
    import torch
    import mamba_ssm
    from mamba_ssm import ops
    import selective_scan_cuda
    # quick smoke test on GPU if available
    if torch.cuda.is_available():
        import torch as T
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        B,D,L,N = 2, 8, 16, 16
        u = T.randn(B,D,L, device='cuda', dtype=T.float16)
        delta = T.rand(B,D,L, device='cuda', dtype=T.float16)
        A = T.randn(D,N, device='cuda', dtype=T.float32)
        Bm = T.randn(D,N, device='cuda', dtype=T.float32)
        Cm = T.randn(D,N, device='cuda', dtype=T.float32)
        y = selective_scan_fn(u, delta, A, Bm, Cm)
        y.sum().backward() if y.requires_grad else None
        print('runtime check: OK on CUDA')
    print('mamba-ssm import: OK, ops/selective_scan_cuda present')
except Exception as e:
    ok = False
    print('mamba-ssm CUDA import failed after upgrade:', e)
sys.exit(0 if ok else 42)
PY
  echo "All done."
else
  if [[ ${CODE} -ne 0 ]]; then
    echo "Build/import still failing. Re-run with PT_UPGRADE=1 to upgrade PyTorch and retry." >&2
    exit 1
  fi
  echo "All done."
fi

# Optional: verify causal-conv1d and rebuild from source if needed
banner "Verify causal-conv1d"
set +e
python - <<'PY'
ok=True
try:
    import causal_conv1d, causal_conv1d_cuda  # noqa
    print('causal-conv1d import: OK')
except Exception as e:
    ok=False
    print('causal-conv1d import failed:', e)
import sys
sys.exit(0 if ok else 42)
PY
CCODE=$?
set -e
if [[ ${CCODE} -ne 0 ]]; then
  banner "Rebuild causal-conv1d from source"
  SRC="${PWD}/third_party/causal-conv1d"
  rm -rf "$SRC" || true
  git clone --depth 1 --branch v1.2.2 https://github.com/Dao-AILab/causal-conv1d.git "$SRC"
  python - <<'PY'
import os
p = os.path.join('third_party','causal-conv1d','setup.py')
s = open(p,'r',encoding='utf-8').read()
s = s.replace('include_dirs=[Path(this_dir) / "csrc" / "causal_conv1d"]',
              'include_dirs=[Path(this_dir)/"csrc"/"causal_conv1d", Path(os.environ.get("CUDA_HOME",""))/"targets"/"x86_64-linux"/"include"]')
if 'allow-unsupported-compiler' not in s:
    s = s.replace('"-O3",', '"-O3","-allow-unsupported-compiler",')
open(p,'w',encoding='utf-8').write(s)
print('Patched', p)
PY
  python -m pip uninstall -y causal-conv1d || true
  python -m pip install --no-build-isolation "$SRC"
  banner "Re-verify causal-conv1d"
  python - <<'PY'
try:
    import causal_conv1d, causal_conv1d_cuda
    print('causal-conv1d: OK after source build')
except Exception as e:
    print('causal-conv1d still failing:', e)
    raise SystemExit(42)
PY
fi
