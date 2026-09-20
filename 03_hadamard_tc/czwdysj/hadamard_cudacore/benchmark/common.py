from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build" / "libhadamard.so"
VERSION_IDS = {"v0": 0, "v1": 1, "v2": 2}
LEGACY_VERSION_NAMES = {"naive": "v0", "shared": "v1", "optimized": "v2"}
DTYPE_IDS = {"fp32": 0, "fp16": 1, "bf16": 2}
TOLERANCES = {"fp32": (2e-5, 1e-4), "fp16": (1e-2, 2e-2), "bf16": (3e-2, 5e-2)}
TEST_PATTERNS = ("random", "basis", "sequence")


def normalize_version(version: str) -> str:
    return LEGACY_VERSION_NAMES.get(version, version)


def torch_dtype(torch: Any, name: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


class PreparedHadamard:
    """One version and one input with output/workspace allocated before timing."""

    def __init__(self, library: "HadamardLibrary", version: str, x: Any, dtype_name: str) -> None:
        version = normalize_version(version)
        if version not in VERSION_IDS:
            raise ValueError(f"unknown kernel version: {version}")
        if dtype_name not in DTYPE_IDS:
            raise ValueError(f"unknown dtype: {dtype_name}")
        if not x.is_cuda or not x.is_contiguous() or x.ndim != 2:
            raise ValueError("input must be a contiguous rank-2 CUDA tensor")
        if x.dtype != torch_dtype(library.torch, dtype_name):
            raise ValueError(f"tensor dtype does not match --dtype {dtype_name}")
        self.library, self.version, self.x, self.dtype_name = library, version, x, dtype_name
        self.batch, self.dim = x.shape
        self.output = library.torch.empty_like(x)
        self.workspace_bytes = library.lib.hadamard_workspace_bytes(
            VERSION_IDS[version], self.batch, self.dim, DTYPE_IDS[dtype_name]
        )
        self.workspace = library.torch.empty(self.workspace_bytes, dtype=library.torch.uint8, device=x.device)

    def launch(self):
        stream = self.library.torch.cuda.current_stream(self.x.device)
        workspace_ptr = self.workspace.data_ptr() if self.workspace_bytes else 0
        error = self.library.lib.hadamard_launch(
            VERSION_IDS[self.version], ctypes.c_void_p(self.x.data_ptr()),
            ctypes.c_void_p(self.output.data_ptr()), ctypes.c_void_p(workspace_ptr),
            self.batch, self.dim, DTYPE_IDS[self.dtype_name], ctypes.c_void_p(stream.cuda_stream)
        )
        if error:
            message = self.library.lib.hadamard_cuda_error_string(error).decode()
            raise RuntimeError(f"hadamard_launch({self.version}) failed: {message}")
        return self.output


class HadamardLibrary:
    def __init__(self, torch: Any, path: Path = DEFAULT_LIBRARY) -> None:
        if not path.exists():
            raise FileNotFoundError(f"CUDA library not found: {path}. Run cmake --build build first.")
        self.torch = torch
        self.lib = ctypes.CDLL(str(path))
        self.lib.hadamard_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        self.lib.hadamard_workspace_bytes.restype = ctypes.c_size_t
        self.lib.hadamard_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.hadamard_launch.restype = ctypes.c_int
        self.lib.hadamard_cuda_error_string.argtypes = [ctypes.c_int]
        self.lib.hadamard_cuda_error_string.restype = ctypes.c_char_p

    def prepare(self, version: str, x: Any, dtype_name: str) -> PreparedHadamard:
        return PreparedHadamard(self, version, x, dtype_name)


def reference_hadamard(torch: Any, x: Any):
    """FP32 iterative Sylvester/Walsh FWHT with orthogonal normalization."""
    batch, dim = x.shape
    values = x.float().clone()
    stride = 1
    while stride < dim:
        view = values.view(batch, dim // (2 * stride), 2, stride)
        low, high = view[:, :, 0, :].clone(), view[:, :, 1, :].clone()
        view[:, :, 0, :], view[:, :, 1, :] = low + high, low - high
        stride *= 2
    return values.mul_(1.0 / math.sqrt(dim))


def error_metrics(torch: Any, actual: Any, expected: Any) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), (difference / expected.float().abs().clamp_min(1e-6)).max().item()


def is_correct(torch: Any, actual: Any, expected: Any, dtype_name: str, version: str | None = None) -> bool:
    # v0 stores each butterfly stage in the destination dtype. BF16 therefore
    # accumulates more rounding at D=8192 than v1/v2, which keep stages in FP32.
    atol, rtol = (8e-2, 8e-2) if (version == "v0" and dtype_name == "bf16") else TOLERANCES[dtype_name]
    return bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))


def make_test_input(torch: Any, pattern: str, batch: int, dim: int, dtype_name: str):
    dtype = torch_dtype(torch, dtype_name)
    if pattern == "random":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260918 + dim)
        return torch.randn((batch, dim), generator=generator, device="cuda", dtype=dtype).contiguous()
    if pattern == "basis":
        x = torch.zeros((batch, dim), device="cuda", dtype=dtype)
        for row, position in enumerate((0, 1, dim // 3, dim - 1)[:batch]):
            x[row, position] = 1
        return x.contiguous()
    if pattern == "sequence":
        base = ((torch.arange(dim, device="cuda") % 17) - 8).float() / 8.0
        return torch.stack([torch.roll(base, shifts=row * 3) for row in range(batch)]).to(dtype).contiguous()
    raise ValueError(f"unknown test pattern: {pattern}")


@dataclass
class PairMetrics:
    max_abs_error: float = 0.0
    max_relative_error: float = 0.0
    passed: bool = True

    def update(self, result: tuple[float, float, bool]) -> None:
        max_abs, max_relative, passed = result
        self.max_abs_error, self.max_relative_error = max(self.max_abs_error, max_abs), max(self.max_relative_error, max_relative)
        self.passed = self.passed and passed


@dataclass
class CorrectnessGate:
    version_vs_reference: PairMetrics
    dao_vs_reference: PairMetrics
    version_vs_dao: PairMetrics

    @property
    def passed(self) -> bool:
        return self.version_vs_reference.passed and self.dao_vs_reference.passed and self.version_vs_dao.passed


def check_pair(torch: Any, actual: Any, expected: Any, dtype_name: str, version: str | None = None) -> tuple[float, float, bool]:
    max_abs, max_relative = error_metrics(torch, actual, expected)
    return max_abs, max_relative, is_correct(torch, actual, expected, dtype_name, version)


def run_correctness_gate(torch: Any, library: HadamardLibrary, hadamard_transform: Callable[..., Any],
                         version: str, batch: int, dim: int, dtype_name: str,
                         patterns: tuple[str, ...] = TEST_PATTERNS) -> CorrectnessGate:
    check_batch, scale = max(1, min(batch, 4)), 1.0 / math.sqrt(dim)
    gate = CorrectnessGate(PairMetrics(), PairMetrics(), PairMetrics())
    for pattern in patterns:
        x = make_test_input(torch, pattern, check_batch, dim, dtype_name)
        prepared = library.prepare(version, x, dtype_name)
        reference, mine = reference_hadamard(torch, x), prepared.launch()
        dao = hadamard_transform(x, scale=scale)
        torch.cuda.synchronize()
        gate.version_vs_reference.update(check_pair(torch, mine, reference, dtype_name, version))
        gate.dao_vs_reference.update(check_pair(torch, dao, reference, dtype_name))
        gate.version_vs_dao.update(check_pair(torch, mine, dao, dtype_name, version))
    return gate
