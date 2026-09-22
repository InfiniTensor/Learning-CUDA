"""Stream-safe Python interface to the small software CUDA codec."""

from dataclasses import dataclass
import ctypes
from pathlib import Path

import torch

DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
LIB = ctypes.CDLL(str(Path(__file__).with_name("libcodec.so")))
PTR = ctypes.c_void_p
INT = ctypes.c_int
LIB.quantize.argtypes = (
    [PTR] * 5 + [INT] * 6 + [ctypes.c_uint, INT, INT, ctypes.c_float, PTR]
)
LIB.quantize.restype = INT
LIB.dequantize.argtypes = [PTR] * 4 + [INT] * 5 + [PTR]
LIB.dequantize.restype = INT


@dataclass
class QuantizedTensor:
    data: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    rows: int
    cols: int
    format: str = "mxfp8"
    scale_mode: str = "block"
    rounding: str = "nearest"
    scale_policy: str = "floor"
    four_over_six: bool = False
    fp8_bound: float = 448.0
    seed: int = 17

    @property
    def nbytes(self):
        return (
            self.data.numel()
            + self.scales.numel()
            + (4 if self.format == "nvfp4" else 0)
        )


def prepare(
    x,
    *,
    format="mxfp8",
    scale_mode="block",
    rounding="nearest",
    scale_policy="floor",
    four_over_six=False,
    fp8_bound=None,
    seed=17,
):
    if x.ndim != 2 or min(x.shape) <= 0 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("expected a nonempty contiguous 2D CUDA tensor")
    if x.dtype not in DTYPES or format not in ("mxfp8", "nvfp4"):
        raise ValueError("unsupported input dtype or format")
    if scale_mode not in ("block", "tensor") or rounding not in (
        "nearest",
        "stochastic",
    ):
        raise ValueError("unsupported scale_mode or rounding")
    if scale_policy not in ("floor", "rceil"):
        raise ValueError("scale_policy must be floor or rceil")
    if four_over_six and (
        format != "nvfp4" or scale_mode != "block" or rounding != "nearest"
    ):
        raise ValueError("4over6 uses block-scaled NVFP4 with nearest rounding")
    rows, cols = x.shape
    block = 16 if format == "nvfp4" else 32
    if rows * ((cols + block - 1) // block) >= 2**31:
        raise ValueError("matrix exceeds the CUDA grid supported by this baseline")
    count = 1 if scale_mode == "tensor" else rows * ((cols + block - 1) // block)
    data_shape = (rows, (cols + 1) // 2) if format == "nvfp4" else (rows, cols)
    bound = float(fp8_bound if fp8_bound is not None else 256 if four_over_six else 448)
    if not 0 < bound <= 448:
        raise ValueError("fp8_bound must be in (0,448]")
    qt = QuantizedTensor(
        torch.empty(data_shape, device=x.device, dtype=torch.uint8),
        torch.empty(count, device=x.device, dtype=torch.uint8),
        torch.empty(1, device=x.device, dtype=torch.float32),
        rows,
        cols,
        format,
        scale_mode,
        rounding,
        scale_policy,
        four_over_six,
        bound,
        seed,
    )
    amax = (
        x.float().abs().amax().reshape(1)
        if format == "nvfp4" or scale_mode == "tensor"
        else torch.zeros(1, device=x.device)
    )
    return qt, amax


def quantize_into(x, qt, amax):
    with torch.cuda.device(x.device):
        err = LIB.quantize(
            x.data_ptr(),
            qt.data.data_ptr(),
            qt.scales.data_ptr(),
            amax.data_ptr(),
            qt.global_scale.data_ptr(),
            qt.rows,
            qt.cols,
            qt.format == "nvfp4",
            DTYPES[x.dtype],
            qt.scale_mode == "tensor",
            qt.rounding == "stochastic",
            qt.seed,
            qt.scale_policy == "rceil",
            qt.four_over_six,
            qt.fp8_bound,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if err:
        raise RuntimeError(f"CUDA quantize launch error {err}")
    return qt


def quantize(x, **kwargs):
    x = x.contiguous()
    qt, amax = prepare(x, **kwargs)
    return quantize_into(x, qt, amax)


def dequantize_into(qt, y):
    if y.shape != (qt.rows, qt.cols) or y.dtype not in DTYPES or not y.is_contiguous():
        raise ValueError("invalid output shape/dtype/layout")
    if y.device != qt.data.device:
        raise ValueError("output and packed data must be on the same CUDA device")
    with torch.cuda.device(y.device):
        err = LIB.dequantize(
            qt.data.data_ptr(),
            qt.scales.data_ptr(),
            qt.global_scale.data_ptr(),
            y.data_ptr(),
            qt.rows,
            qt.cols,
            qt.format == "nvfp4",
            DTYPES[y.dtype],
            qt.scale_mode == "tensor",
            torch.cuda.current_stream(y.device).cuda_stream,
        )
    if err:
        raise RuntimeError(f"CUDA dequantize launch error {err}")
    return y


def dequantize(qt, dtype=torch.float32):
    return dequantize_into(
        qt, torch.empty((qt.rows, qt.cols), dtype=dtype, device=qt.data.device)
    )
