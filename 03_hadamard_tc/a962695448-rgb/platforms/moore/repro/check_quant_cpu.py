"""Regenerate the exact INT4 fixture using the independently checked CPU reference."""

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--library",
    type=Path,
    required=True,
    help="Path to the compiled CPU comparison library.",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="New output directory for the fixture and report.",
)
args = parser.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
OUTPUT = args.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
scan = lib.scan_quant
scan.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
]
scan.restype = ctypes.c_size_t
count = 0
gpu_values, gpu_scales = [], []


def check(values, scales, keep=False):
    global count
    values = np.ascontiguousarray(values, dtype=np.float32)
    scales = np.ascontiguousarray(
        np.broadcast_to(scales, values.shape), dtype=np.float32
    )
    first = np.zeros(5, dtype=np.uint32)
    errors = scan(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        first.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    if errors:
        raise AssertionError(
            {
                "errors": errors,
                "first": first.tolist(),
                "value": float(values[first[0]]),
                "scale": float(scales[first[0]]),
            }
        )
    count += values.size
    if keep:
        gpu_values.append(values.copy())
        gpu_scales.append(scales.copy())


started = time.monotonic()
per_dtype = {}
for name, maximum in [("fp16", 0x7BFF), ("bf16", 0x7F7F)]:
    bits = np.arange(maximum + 1, dtype=np.uint16)
    pool = (
        bits.view(np.float16).astype(np.float32)
        if name == "fp16"
        else (bits.astype(np.uint32) << 16).view(np.float32)
    )
    indices = np.unique(np.linspace(1, maximum, 257).astype(int))
    gpu_indices = set(np.unique(np.linspace(1, maximum, 17).astype(int)))
    before = count
    for limit in sorted(set(indices) | gpu_indices):
        scale = np.float32(pool[limit] / np.float32(7))
        values = pool[: limit + 1]
        check(values, scale, limit in gpu_indices)
        check(-values, scale, limit in gpu_indices)
    per_dtype[name] = count - before

# Probe the composite binary32 quotient/integer rounding boundary from both sides.
rng = np.random.default_rng(20260916)
scales = rng.integers(1, 0x7E000000, size=10000, dtype=np.uint32).view(np.float32)
for integer in range(7):
    middle = np.float32(scales * np.float32(integer + 0.5))
    values = middle.copy()
    for offset in range(4):
        check(values, scales, True)
        check(-values, scales, True)
        values = np.nextafter(values, np.float32(np.inf))
    values = middle.copy()
    for offset in range(3):
        values = np.nextafter(values, np.float32(0))
        check(values, scales, True)
        check(-values, scales, True)

values = np.concatenate(gpu_values)
scales = np.concatenate(gpu_scales)
expected = np.clip(np.rint(np.float32(values / scales)), -7, 7).astype(np.int32)
dtype = np.dtype([("value", "<f4"), ("scale", "<f4"), ("expected", "<i4")])
fixture = np.empty(values.size, dtype=dtype)
fixture["value"], fixture["scale"], fixture["expected"] = values, scales, expected
path = OUTPUT / "quant_fixture.bin"
fixture.tofile(path)
result = {
    "status": "PASS",
    "cpu_cases": count,
    "per_dtype": per_dtype,
    "boundary_cases": count - sum(per_dtype.values()),
    "gpu_fixture_cases": values.size,
    "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "header_sha256": hashlib.sha256(
        (ROOT.parent / "exact_int4.hpp").read_bytes()
    ).hexdigest(),
    "seconds": time.monotonic() - started,
    "numpy_version": np.__version__,
    "scope": "Actual C++ bit-comparison implementation versus existing CPU divide/round reference; this is not a GPU test",
}
(OUTPUT / "cpu-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
