"""Portable little-endian JSON header plus packed binary payload, version 1."""

from dataclasses import fields
import json
from pathlib import Path
import struct

import torch
from codec import QuantizedTensor

TENSOR_MAGIC = b"LPTENS1\0"
QUANT_MAGIC = b"LPQUAN1\0"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def write_header(f, magic, header):
    data = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    f.write(magic)
    f.write(struct.pack("<I", len(data)))
    f.write(data)


def read_header(f, magic):
    if f.read(8) != magic:
        raise ValueError("bad magic/version")
    length_bytes = f.read(4)
    if len(length_bytes) != 4:
        raise ValueError("truncated header")
    length = struct.unpack("<I", length_bytes)[0]
    if length > 1024 * 1024:
        raise ValueError("header exceeds 1 MiB")
    return json.loads(f.read(length))


def tensor_bytes(t):
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def save_tensor(path, x):
    if x.ndim != 2:
        raise ValueError("expected matrix")
    name = next(n for n, d in DTYPES.items() if d == x.dtype)
    with Path(path).open("wb") as f:
        write_header(
            f, TENSOR_MAGIC, dict(num_rows=x.shape[0], num_cols=x.shape[1], dtype=name)
        )
        f.write(tensor_bytes(x))


def load_tensor(path, device="cpu"):
    with Path(path).open("rb") as f:
        h = read_header(f, TENSOR_MAGIC)
        raw = bytearray(f.read())
    r, c = int(h["num_rows"]), int(h["num_cols"])
    dtype = DTYPES[h["dtype"]]
    if (
        r <= 0
        or c <= 0
        or len(raw) != r * c * torch.empty((), dtype=dtype).element_size()
    ):
        raise ValueError("tensor payload size mismatch")
    return torch.frombuffer(raw, dtype=dtype).reshape(r, c).clone().to(device)


def save_quantized(path, qt):
    h = {
        f.name: getattr(qt, f.name)
        for f in fields(qt)
        if f.name not in ("data", "scales", "global_scale")
    }
    h.update(
        data_bytes=qt.data.numel(),
        scale_bytes=qt.scales.numel(),
        global_scale_bytes=4 if qt.format == "nvfp4" else 0,
        element_order="row-major",
        nibble_order="even-low-odd-high",
    )
    with Path(path).open("wb") as f:
        write_header(f, QUANT_MAGIC, h)
        f.write(tensor_bytes(qt.data))
        f.write(tensor_bytes(qt.scales))
        if qt.format == "nvfp4":
            f.write(tensor_bytes(qt.global_scale))


def load_quantized(path, device="cuda"):
    with Path(path).open("rb") as f:
        h = read_header(f, QUANT_MAGIC)
        raw = bytearray(f.read())
    r, c = int(h["rows"]), int(h["cols"])
    fp4 = h["format"] == "nvfp4"
    if h["format"] not in ("mxfp8", "nvfp4") or h["scale_mode"] not in (
        "block",
        "tensor",
    ):
        raise ValueError("unsupported format/scale_mode")
    if h["global_scale_bytes"] != (4 if fp4 else 0):
        raise ValueError("invalid global scale size")
    n = r * ((c + 1) // 2) if fp4 else r * c
    size = 16 if fp4 else 32
    ns = 1 if h["scale_mode"] == "tensor" else r * ((c + size - 1) // size)
    if (
        r <= 0
        or c <= 0
        or n != h["data_bytes"]
        or ns != h["scale_bytes"]
        or len(raw) != n + ns + (4 if fp4 else 0)
    ):
        raise ValueError("quantized payload size mismatch")
    q = (
        torch.frombuffer(raw, dtype=torch.uint8, count=n)
        .clone()
        .reshape(r, -1)
        .to(device)
    )
    s = torch.frombuffer(raw, dtype=torch.uint8, count=ns, offset=n).clone().to(device)
    g = (
        torch.frombuffer(raw, dtype=torch.float32, count=1, offset=n + ns)
        .clone()
        .to(device)
        if fp4
        else torch.ones(1, device=device)
    )
    cfg = {f.name: h[f.name] for f in fields(QuantizedTensor) if f.name in h}
    return QuantizedTensor(q, s, g, **cfg)
