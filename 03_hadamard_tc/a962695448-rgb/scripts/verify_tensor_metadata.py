#!/usr/bin/env python3
"""验证惰性负号视图明确拒绝，以及显式物化后的三种接口。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from build_torch_extension import load_extension


def check_metadata(torch, op):
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for shape in ((3, 256), (1, 1, 3, 256)):
            base = torch.arange(768, device="cuda", dtype=torch.float32).remainder(29).sub(14).div(16).to(dtype).reshape(shape)
            before = base.clone()
            lazy = torch._neg_view(base)
            if not lazy.is_neg() or not lazy.is_contiguous() or lazy.data_ptr() != base.data_ptr():
                raise RuntimeError("fixture does not exercise a contiguous lazy negative alias")
            materialized, expected_input = lazy.resolve_neg(), -base
            if materialized.is_neg() or not torch.equal(materialized.view(torch.uint8), expected_input.view(torch.uint8)):
                raise RuntimeError("resolve_neg fixture did not materialize the logical values")
            for threads in (128, 256):
                for method in ("hadamard", "hadamard_int4", "quantize_int4"):
                    operation = getattr(op, method)
                    try:
                        operation(lazy, block_threads=threads)
                    except RuntimeError as error:
                        if "resolve_neg" not in str(error):
                            raise
                        reason = str(error).splitlines()[0]
                    else:
                        raise RuntimeError(f"accepted lazy negative alias: {method}/{dtype}/{shape}/{threads}")
                    actual = operation(materialized, block_threads=threads)
                    expected = operation(expected_input, block_threads=threads)
                    actual = actual if isinstance(actual, tuple) else (actual,)
                    expected = expected if isinstance(expected, tuple) else (expected,)
                    if len(actual) != len(expected) or any(a.shape != b.shape or a.dtype != b.dtype or
                            not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected)):
                        raise RuntimeError("materialized logical input output mismatch: " + method)
                    results.append({"dtype": str(dtype), "shape": list(shape), "block_threads": threads,
                                    "method": method, "rejected_lazy_alias": True,
                                    "materialized_output_bitwise_equal": True, "error": reason})
            try:
                op.hadamard_int4(lazy, fused_layout="contiguous256")
            except RuntimeError as error:
                if "resolve_neg" not in str(error):
                    raise
                reason = str(error).splitlines()[0]
            else:
                raise RuntimeError("contiguous256 fused accepted a lazy negative alias")
            actual = op.hadamard_int4(materialized, fused_layout="contiguous256")
            expected = op.hadamard_int4(expected_input)
            if len(actual) != len(expected) or any(a.shape != b.shape or a.dtype != b.dtype or
                    not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected)):
                raise RuntimeError("materialized contiguous256 fused differs from original layout")
            results.append({"dtype": str(dtype), "shape": list(shape), "block_threads": 128,
                            "method": "hadamard_int4", "fused_layout": "contiguous256",
                            "rejected_lazy_alias": True, "materialized_output_bitwise_equal": True,
                            "error": reason})
            if not torch.equal(base.view(torch.uint8), before.view(torch.uint8)):
                raise RuntimeError("underlying input was modified")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("result already exists; choose a new output path")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        op = load_extension(verbose=True, build_directory=str(args.build_directory))
        report["environment"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(),
                                 "extension_sha256": hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest()}
        report["cases"] = check_metadata(torch, op)
        report["status"] = "PASS"
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("src/torch_binding.cu", "include/kernels.cuh", "include/contiguous256.cuh",
                     "scripts/verify_tensor_metadata.py")}
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metadata_cases": len(report.get("cases", []))}))
    if report["status"] != "PASS":
        print(report["traceback"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
