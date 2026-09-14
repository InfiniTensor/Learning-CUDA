'''这个文件用来写量化格式'''
import random

MAX_FENMU = {"nvfp4": 6.0, "mxfp8": 448.0}
# fp4就16个数，直接查表就行
_E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# fp8也查表，但是数太多了
_E4M3_TABLE = []
# 128个正数包括0一起
for code in range(128): # 1111 110
    # 移除尾数，得到指数
    e = (code >> 3) & 0xF
    m = code & 0x07
    if e == 15 and m == 7:
        val = 448.0
    elif e == 0: 
        # 次正规数
        val = (m/8.0) * (2.0 ** -6)
        # 正规值
    else: val = (1.0 + m/8.0) * (2.0 ** (e-7))
    _E4M3_TABLE.append(val)

def float_to_e2m1(x, rng):
    if x == 0.0:
        return 0
    a = abs(x)
    low_idx = 0
    for i in range(1, 8):
        if a >= _E2M1_TABLE[i]: low_idx = i
    high_idx = min(low_idx+1, 7)
    low_val = _E2M1_TABLE[low_idx]
    high_val = _E2M1_TABLE[high_idx]

    if rng is not None:
        # stochastic
        pi = (a - low_val) / (high_val - low_val)
        idx = high_idx if rng.random() < pi else low_idx
    else:
        # nearest：离谁近选谁
        idx = high_idx if (a - low_val) >= (high_val - a) else low_idx
    return idx if x > 0 else idx + 8

def float_to_e4m3(x, rng):
    if x == 0.0:
        return 0

    sign = 0x80 if x < 0 else 0
    a = abs(x)
    low_idx = 0
    for i in range(1, 128):
        if a >= _E4M3_TABLE[i]: low_idx = i
    high_idx = min(low_idx+1, 127)
    low_val = _E4M3_TABLE[low_idx]
    high_val = _E4M3_TABLE[high_idx]

    if rng is not None:
        pi = (a - low_val) / (high_val - low_val)
        idx = high_idx if rng.random() < pi else low_idx
    else:
        idx = high_idx if (a - low_val) > (high_val - a) else low_idx

    return sign | idx

def pack(codes, element_bits):
    if element_bits == 8: return bytes(codes)
    if len(codes) % 2 == 1:
        codes = codes + [0]          # 奇数个就补一个 0
    out = bytearray()
    for i in range(0, len(codes), 2):
        out.append((codes[i] & 0xF) | ((codes[i + 1] & 0xF) << 4))
    return bytes(out)

def unpack(packed, element_bits, numbers):
    if element_bits == 8:
        return list(packed[:numbers])
    codes = []
    for byte in packed:
        codes.append(byte & 0xF)
        codes.append((byte >> 4) & 0xF)
    return codes[:numbers]

def e2m1_to_float(idx):
    return _E2M1_TABLE[idx]

def e4m3_to_float(idx):
    val = _E4M3_TABLE[idx & 0x7F]
    return -val if (idx & 0x80) else val
