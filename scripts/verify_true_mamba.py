#!/usr/bin/env python
import sys

def main() -> int:
    try:
        import torch
        print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    except Exception as e:
        print("Torch import failed:", e)
        return 2

    try:
        import mamba_ssm
        from mamba_ssm import ops
        import selective_scan_cuda  # compiled CUDA backend
        print("mamba-ssm: OK; selective_scan_cuda present")
    except Exception as e:
        print("mamba-ssm CUDA import failed:", e)
        return 1

    # Quick functional smoke test on CUDA
    try:
        if torch.cuda.is_available():
            import torch as T
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            B, D, L, N = 1, 4, 8, 8
            u = T.randn(B, D, L, device='cuda', dtype=T.float16, requires_grad=True)
            delta = T.rand(B, D, L, device='cuda', dtype=T.float16)
            A = T.randn(D, N, device='cuda', dtype=T.float32)
            Bm = T.randn(D, N, device='cuda', dtype=T.float32)
            Cm = T.randn(D, N, device='cuda', dtype=T.float32)
            out = selective_scan_fn(u, delta, A, Bm, Cm)
            out.sum().backward()
            print("runtime: OK on CUDA (forward+backward)")
        else:
            print("CUDA not available; import-only verification done.")
    except Exception as e:
        print("runtime check failed:", e)
        return 3

    print("SUCCESS: Real mamba-ssm CUDA extension is active.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

