'''这个文件用来写量化格式'''

MAX_FENMU = {"nvfp4": 6.0, "mxfp8": 448.0}
# fp4就16个数，直接查表就行
_E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def float_to_e2m1(x, rng):
    if rng is not None:
        pass

    if x == 0.0:
        return 0
    a = abs(x)
    if   a >= 5.0:  idx = 7
    elif a >= 3.5:  idx = 6
    elif a >= 2.5:  idx = 5
    elif a >= 1.75: idx = 4
    elif a >= 1.25: idx = 3
    elif a >= 0.75: idx = 2
    elif a >= 0.25: idx = 1
    else:           idx = 0 
    return idx if x > 0 else idx + 8

def float_to_e4m3(x, rng):
    pass

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

def e4m3_to_float(val):
    pass