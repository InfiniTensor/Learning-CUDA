"""Independent NumPy oracle and GPU PyTorch baseline for codec tests."""

import numpy as np

E2 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
E4 = np.array(
    [
        np.ldexp(float(i & 7), -9)
        if i < 8
        else np.ldexp(float(8 + (i & 7)), (i >> 3) - 10)
        for i in range(127)
    ],
    dtype=np.float32,
)


def encode_numpy(x, lut, sign_bit):
    a = np.abs(x)
    hi = np.searchsorted(lut, a, side="left").clip(0, len(lut) - 1)
    lo = np.maximum(hi - 1, 0)
    dl, dh = a - lut[lo], lut[hi] - a
    code = np.where((dh < dl) | ((dh == dl) & (hi % 2 == 0)), hi, lo)
    return code.astype(np.uint8) | (np.signbit(x).astype(np.uint8) << sign_bit)


def numpy_reference(
    x,
    format="mxfp8",
    scale_mode="block",
    scale_policy="floor",
    four_over_six=False,
    fp8_bound=None,
):
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    size = 16 if format == "nvfp4" else 32
    group_count = (cols + size - 1) // size
    padded = np.pad(x, ((0, 0), (0, group_count * size - cols))).reshape(
        rows, group_count, size
    )
    a = np.max(np.abs(padded), axis=-1)
    if scale_mode == "tensor":
        a[:] = np.max(np.abs(x))
    if format == "mxfp8":
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.floor(np.log2(a)) - 8
            if scale_policy == "rceil":
                e = np.ceil(np.log2(a / np.float32(448)))
            e = np.where(a == 0, -127, e).clip(-127, 127).astype(np.int32)
        scale = np.ldexp(np.ones_like(a), e)
        q = encode_numpy(padded / scale[..., None], E4, 7)
        y = np.ldexp(
            np.copysign(E4[q & 127], np.where(q & 128, -1, 1)), e[..., None]
        ).astype(np.float32)
        s = (e + 127).astype(np.uint8)
        data = q.reshape(rows, -1)[:, :cols].copy()
        g = np.float32(1)
    else:
        bound = np.float32(
            fp8_bound if fp8_bound is not None else 256 if four_over_six else 448
        )
        at = np.max(np.abs(x))
        enc = np.float32(np.float32(6 * bound) / at) if at else np.float32(1)
        g = (
            np.float32(max(np.float32(1) / enc, np.finfo(np.float32).tiny))
            if at
            else np.float32(1)
        )

        def candidate(m):
            sc = (a / np.float32(6)) * enc
            if m == 4:
                sc = sc * np.float32(1.5)
            s = encode_numpy(sc, E4, 7)
            d = E4[s] * g
            inv = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
            scaled = padded * inv[..., None]
            q = encode_numpy(scaled, E2, 3)
            values = np.copysign(E2[q & 7], np.where(q & 8, -1, 1)).astype(np.float32)
            y = (values * E4[s][..., None]) * g
            return q, s, y

        q, s, y = candidate(6)
        if four_over_six:
            q4, s4, y4 = candidate(4)
            better = np.sum((padded - y4) ** 2, axis=-1) < np.sum(
                (padded - y) ** 2, axis=-1
            )
            q = np.where(better[..., None], q4, q)
            s = np.where(better, s4, s)
            y = np.where(better[..., None], y4, y)
        q = q.reshape(rows, -1)[:, :cols]
        if cols % 2:
            q = np.pad(q, ((0, 0), (0, 1)))
        data = q[:, ::2] | (q[:, 1::2] << 4)
    s = s.reshape(-1)[:1] if scale_mode == "tensor" else s.reshape(-1)
    return data, s, g, y.reshape(rows, -1)[:, :cols].copy()


def metrics(x, y):
    a = x.double()
    d = y.double() - a
    sse = d.square().sum().item()
    norm = a.square().sum().item()
    return dict(
        max_abs=d.abs().max().item(),
        mae=d.abs().mean().item(),
        mse=sse / x.numel(),
        rel_l2=(sse / norm) ** 0.5 if norm else 0,
        squared_error_sum=sse,
        input_squared_sum=norm,
    )
