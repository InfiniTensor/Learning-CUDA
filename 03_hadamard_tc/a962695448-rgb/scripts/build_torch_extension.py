#!/usr/bin/env python3
"""Build/load the forward-only PyTorch adapter using the current PyTorch ABI."""
import argparse
import json
import os
from pathlib import Path


def load_extension(verbose=False, build_directory=None):
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and a visible NVIDIA GPU are required")
    root = Path(__file__).resolve().parents[1]
    build = Path(build_directory) if build_directory else root / "build" / "torch_extension"
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    # PyTorch selects the visible device's architecture unless the caller explicitly
    # sets TORCH_CUDA_ARCH_LIST (e.g. 8.9 for RTX4090, 8.0 for A100).
    return load(
        name="infinitensor_hadamard_cuda",
        sources=[str(root / "src" / "torch_binding.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-lineinfo",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build),
        with_cuda=True,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--build-directory")
    args = parser.parse_args()
    try:
        import torch
        extension = load_extension(args.verbose, args.build_directory)
        sample = torch.tensor([[3.0, 1.0]], device="cuda", dtype=torch.float16)
        actual = extension.hadamard(sample)
        torch.cuda.synchronize()
        if actual.cpu().tolist() != [[4.0, 2.0]]:
            raise RuntimeError(f"extension smoke failed: {actual.cpu().tolist()}")
        print(json.dumps({"status": "PASS", "torch": torch.__version__,
                          "torch_cuda": torch.version.cuda,
                          "module": str(Path(extension.__file__).resolve()),
                          "gpu": torch.cuda.get_device_name(), "smoke": [[4.0, 2.0]]}))
        return 0
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error": f"{type(error).__name__}: {error}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
