#!/usr/bin/env python3
import os, sys

def main() -> int:
    print("Seccomp assumed disabled in elevated run.")
    try:
        import torch
    except Exception as e:
        print("torch import failed:", e)
        return 2
    print("torch:", torch.__version__, "cuda_build:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return 1
    print("device0:", torch.cuda.get_device_name(0))
    # Simple matmul smoke test
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    c = a @ b
    print("matmul ok: ", bool(c.isfinite().all().item()))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

