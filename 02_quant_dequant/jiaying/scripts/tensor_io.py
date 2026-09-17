"""Documented little-endian tensor/weight reader; no project C++ code is used."""
from pathlib import Path
import struct
import numpy as np

TENSOR_HEADER = struct.Struct("<8sQQII")
WEIGHT_HEADER = struct.Struct("<8sIIQQIIIIIIQQQfIQ")


def write_tensor(path, values, dtype="fp32"):
    values = np.asarray(values)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Expected a nonempty matrix")
    dt, code = ("<f4", 0) if dtype == "fp32" else ("<f2", 1)
    a = values.astype(dt)
    Path(path).write_bytes(TENSOR_HEADER.pack(b"LPTENS1\0", *a.shape, code, 0) + a.tobytes())
    return a.astype(np.float32)


def read_tensor(path):
    raw = Path(path).read_bytes()
    magic, rows, cols, dtype, flags = TENSOR_HEADER.unpack_from(raw)
    if magic != b"LPTENS1\0" or flags or dtype not in (0, 1, 2):
        raise ValueError("Invalid tensor header")
    data = np.frombuffer(raw, dtype=("<f4", "<f2", "<u2")[dtype], offset=32)
    if data.size != rows * cols:
        raise ValueError("Invalid tensor length")
    if dtype == 2:
        data = (data.astype(np.uint32) << 16).view(np.float32)
    return data.reshape(rows, cols).astype(np.float32)


def read_weights(path):
    raw = Path(path).read_bytes()
    h = WEIGHT_HEADER.unpack_from(raw)
    (magic, version, header_size, rows, cols, fmt, block, mode, out, rounding,
     input_dtype, seed, nscale, ndata, global_scale, flags, reserved) = h
    if magic != b"LPWGHT1\0" or version != 1 or header_size != 96 or flags or reserved:
        raise ValueError("Invalid weights header")
    if len(raw) != 96 + ndata + nscale:
        raise ValueError("Invalid weights length")
    return dict(rows=rows, cols=cols, fmt=fmt, block=block, mode=mode, out=out,
                rounding=rounding, input_dtype=input_dtype, seed=seed,
                global_scale=np.float32(global_scale),
                data=np.frombuffer(raw, np.uint8, count=ndata, offset=96).copy(),
                scales=np.frombuffer(raw, np.uint8, count=nscale, offset=96 + ndata).copy())
