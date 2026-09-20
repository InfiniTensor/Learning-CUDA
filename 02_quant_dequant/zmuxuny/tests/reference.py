"""Independent NumPy/file reference; encoders search explicit codebooks."""

import struct
from pathlib import Path
import numpy as np

TENSOR = struct.Struct("<8sQQII")
PACKED = struct.Struct("<8sQQIIIfQQ")
FP8 = np.array(
    [
        m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        for e in range(16)
        for m in range(8)
    ],
    dtype=np.float32,
)[:127]
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def to_bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & 1)) >> 16).astype("<u2")


def cast(x, dtype):
    if dtype == 0:
        return np.asarray(x, dtype=np.float32)
    if dtype == 1:
        return np.asarray(x, dtype=np.float16).astype(np.float32)
    return (to_bf16(x).astype(np.uint32) << 16).view(np.float32)


def write_tensor(path, x, dtype=0):
    x = np.asarray(x)
    payload = (
        x.astype("<f4") if dtype == 0 else x.astype("<f2") if dtype == 1 else to_bf16(x)
    )
    Path(path).write_bytes(
        TENSOR.pack(b"LPTENS1\0", *x.shape, dtype, 0) + payload.tobytes()
    )


def read_tensor(path):
    data = Path(path).read_bytes()
    magic, r, c, dtype, _ = TENSOR.unpack_from(data)
    assert magic == b"LPTENS1\0" and len(data) == 32 + r * c * (4 if dtype == 0 else 2)
    x = np.frombuffer(
        data, "<f4" if dtype == 0 else "<f2" if dtype == 1 else "<u2", offset=32
    )
    if dtype == 2:
        x = (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32).reshape(r, c), dtype


def read_packed(path):
    data = Path(path).read_bytes()
    magic, r, c, fmt, block, tensor, global_scale, ns, nb = PACKED.unpack_from(data)
    assert magic == b"LPPACK1\0" and len(data) == 56 + ns + nb
    return dict(
        rows=r,
        cols=c,
        fmt=fmt,
        block=block,
        tensor=bool(tensor),
        global_scale=np.float32(global_scale),
        scales=np.frombuffer(data, np.uint8, ns, 56),
        data=np.frombuffer(data, np.uint8, nb, 56 + ns),
    )


def uniform(indices, seed):
    indices = np.asarray(indices, np.uint64)
    x = (
        indices.astype(np.uint32)
        ^ ((indices >> 32).astype(np.uint32) * np.uint32(0x9E3779B9))
        ^ np.uint32(seed)
    )
    x ^= x >> 16
    x *= np.uint32(0x7FEB352D)
    x ^= x >> 15
    x *= np.uint32(0x846CA68B)
    x ^= x >> 16
    return (x >> 8).astype(np.float32) * np.float32(2**-24)


def encode(x, book, stochastic=False, random=0):
    x = np.asarray(x, np.float32)
    a = np.minimum(np.abs(x), book[-1])
    hi = np.searchsorted(book, a, side="left").clip(0, len(book) - 1)
    lo = (hi - 1).clip(0)
    lv, hv = book[lo], book[hi]
    if stochastic:
        p = np.divide(a - lv, hv - lv, out=np.zeros_like(a), where=hv != lv)
        up = np.asarray(random) < p
    else:
        dl, dh = a - lv, hv - a
        up = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    code = np.where(up, hi, lo).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) * (128 if len(book) == 127 else 8))


def quantize(x, fmt, block, tensor=False, stochastic=False, seed=42):
    x = np.asarray(x, np.float32)
    rows, cols = x.shape
    groups = (cols + block - 1) // block
    if tensor:
        maxima = np.array([np.abs(x).max()], np.float32)
        ids = np.zeros(x.shape, np.int64)
    else:
        padded = np.pad(np.abs(x), ((0, 0), (0, groups * block - cols)))
        maxima = padded.reshape(rows, groups, block).max(axis=-1).reshape(-1)
        ids = np.arange(rows)[:, None] * groups + np.arange(cols)[None, :] // block
    global_scale = np.float32(1)
    if fmt == 0:
        # Integer exponent search instead of the CUDA frexp implementation.
        required = maxima.astype(np.float64) / 448
        exp = np.ceil(np.log2(np.where(required == 0, 1, required))).clip(-127, 127)
        scales = (exp + 127).astype(np.uint8)
        scale_values = np.exp2(exp).astype(np.float32)
        scaled = x / scale_values[ids]
    else:
        amax = np.abs(x).max()
        global_scale = (
            np.float32(1)
            if amax == 0
            else np.maximum(np.float32(amax / np.float32(2688)), np.float32(2**-126))
        )
        scales = encode((maxima / global_scale) / np.float32(6), FP8)
        scale_values = FP8[scales]
        scaled = np.divide(
            x / global_scale,
            scale_values[ids],
            out=np.zeros_like(x),
            where=scale_values[ids] != 0,
        )
    codes = encode(
        scaled,
        FP8 if fmt == 0 else FP4,
        stochastic,
        uniform(np.arange(x.size).reshape(x.shape), seed),
    )
    if fmt == 0:
        data = codes.reshape(-1)
    else:
        codes = np.pad(codes, ((0, 0), (0, cols % 2)))
        data = (codes[:, ::2] | (codes[:, 1::2] << 4)).reshape(-1)
    return dict(
        rows=rows,
        cols=cols,
        fmt=fmt,
        block=block,
        tensor=tensor,
        global_scale=global_scale,
        scales=scales,
        data=data,
    )


def dequantize(q):
    rows, cols = q["rows"], q["cols"]
    if q["fmt"] == 0:
        codes = q["data"].reshape(rows, cols)
        values = FP8[codes & 127] * np.where(codes & 128, np.float32(-1), np.float32(1))
        scale_values = np.exp2(q["scales"].astype(np.int32) - 127).astype(np.float32)
    else:
        packed = q["data"].reshape(rows, (cols + 1) // 2)
        codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, -1)[
            :, :cols
        ]
        values = FP4[codes & 7] * np.where(codes & 8, np.float32(-1), np.float32(1))
        scale_values = FP8[q["scales"]]
    groups = (cols + q["block"] - 1) // q["block"]
    ids = (
        0
        if q["tensor"]
        else np.arange(rows)[:, None] * groups + np.arange(cols)[None, :] // q["block"]
    )
    return (values * scale_values[ids]) * q["global_scale"]


def hadamard(x, normalize=True, random_sign=False, seed=7):
    x = np.array(x, dtype=np.float64, copy=True)
    if random_sign:
        x *= np.where(uniform(np.arange(x.shape[1]), seed) < 0.5, -1.0, 1.0)
    d = x.shape[1]
    # Dense Sylvester matrix is an independent oracle for the small test cases.
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return (x @ h) * (1 / np.sqrt(d) if normalize else 1)


def assert_packed(actual, expected):
    for field in ("rows", "cols", "fmt", "block", "tensor", "global_scale"):
        assert actual[field] == expected[field], (field, actual[field], expected[field])
    for field in ("scales", "data"):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
