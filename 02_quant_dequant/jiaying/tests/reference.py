"""Independent oracle: explicit codebooks and searchsorted, not bit truncation."""
import numpy as np

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FP8 = np.array([m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0**(e - 7)
                for e in range(16) for m in range(8)][:127], dtype=np.float32)


def random_unit(indices, seed):
    mask = (1 << 64) - 1
    result = []
    for i in np.asarray(indices).ravel():
        z = (int(i) + seed + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        result.append((z >> 40) / 2**24)
    return np.array(result, np.float32).reshape(np.shape(indices))


def encode(values, table, stochastic=False, indices=None, seed=42):
    values = np.asarray(values, np.float32)
    a = np.minimum(np.abs(values), table[-1])
    upper = np.searchsorted(table, a, side="right").clip(1, len(table) - 1)
    lower = upper - 1
    fraction = (a - table[lower]) / (table[upper] - table[lower])
    up = random_unit(indices, seed) < fraction if stochastic else (
        (fraction > .5) | ((fraction == .5) & ((lower & 1) == 1)))
    code = (lower + up).astype(np.uint8)
    return code | (np.signbit(values).astype(np.uint8) * (8 if len(table) == 8 else 128))


def quantize(a, fmt, block, tensor, stochastic, seed):
    a = np.asarray(a, np.float32)
    rows, cols = a.shape
    nv = fmt == "nvfp4"
    maximum = np.abs(a).max()
    global_scale = (np.float32(1) if maximum == 0 else
                    max(np.float32(maximum / np.float32(2688)), np.nextafter(np.float32(0), np.float32(1)))) if nv else np.float32(1)
    codes = np.zeros_like(a, dtype=np.uint8)
    restored = np.zeros_like(a)
    scales = []
    spans = [(0, a.size)] if tensor else [(r * cols + c, min(r * cols + c + block, (r + 1) * cols))
                                        for r in range(rows) for c in range(0, cols, block)]
    for begin, end in spans:
        values = a.ravel()[begin:end]
        amax = np.abs(values).max()
        if nv:
            sc = encode(np.array([(amax / global_scale) / np.float32(6)]), FP8)[0]
            scale = FP8[sc]
            y = np.copysign(np.zeros_like(values), values) if scale == 0 else (values / global_scale) / scale
        else:
            exponent = int(np.frexp(amax)[1]) - 1 - 8 if amax != 0 else 0
            exponent = min(127, max(-127, exponent))
            sc = exponent + 127
            scale = np.float32(2.0**exponent)
            y = values / scale
        q = encode(y, FP4 if nv else FP8, stochastic, np.arange(begin, end), seed)
        codes.ravel()[begin:end] = q
        mag = (FP4[q & 7] if nv else FP8[q & 127])
        signed = np.copysign(mag, np.where(q & (8 if nv else 128), -1., 1.).astype(np.float32))
        restored.ravel()[begin:end] = (signed * scale) * global_scale
        scales.append(sc)
    if nv:
        padded = np.zeros((rows, cols + (cols & 1)), np.uint8)
        padded[:, :cols] = codes
        data = (padded[:, ::2] | (padded[:, 1::2] << 4)).ravel()
    else:
        data = codes.ravel()
    return data, np.array(scales, np.uint8), global_scale, restored


def output_bytes(a, dtype):
    if dtype == "fp32": return a.astype("<f4").tobytes()
    if dtype == "fp16": return a.astype("<f2").tobytes()
    u = a.astype("<f4").view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
