#!/usr/bin/env python3
"""
hadamard_reference.py: Hadamard 变换算子官方实现，用于验证算子正确性

目标：
本项目验收以官方 `fast_hadamard_transform` 库输出为标准：
  - FP16 容差: max_abs_error < 1e-2
  - BF16 容差: max_abs_error < 5e-2

行为约定：
  1. 默认非归一化: y = H·x (scale=1.0)。若需归一化，必须显式传入 scale = 1/sqrt(head_dim)。
  2. 矩阵顺序: 等同于 scipy.linalg.hadamard(dim) 的 Sylvester 自然序。
  3. 精度策略: 内部使用 FP32 累加，输入/输出保持 FP16/BF16。
  4. 维度陷阱: 变换严格按最后一维进行。输入必须 reshape(-1, head_dim)，传 1D 张量会导致全量数据被当作单行处理，结果完全错误。

"""

import math
import struct
from array import array

# ---------------------------------------------------------------------------
# 低精度二进制编解码 (纯数据 I/O 工具)
# 目标: 在 Python float (FP32/FP64) 与底层 CUDA 字节流 (FP16/BF16) 间无损转换
# ---------------------------------------------------------------------------

def decode_fp16(raw: bytes) -> array:
    """将小端序 FP16 字节流解码为 Python float32 数组。"""
    n = len(raw) // 2
    return array("f", struct.unpack(f"<{n}e", raw))


def decode_bf16(raw: bytes) -> array:
    """将小端序 BF16 字节流解码为 Python float32 数组。bf16 = fp32 高 16 位，解码无损。"""
    n = len(raw) // 2
    out = bytearray(4 * n)
    out[2::4] = raw[0::2]
    out[3::4] = raw[1::2]
    vals = array("f")
    vals.frombytes(bytes(out))
    return vals


def encode_fp16(values) -> bytes:
    """将 Python 浮点数序列打包为小端序 FP16 字节流。"""
    return struct.pack(f"<{len(values)}e", *values)


def _f32_to_bf16_bits(x: float) -> int:
    """将单个 float32 转换为 BF16 的 16 位整数表示。
    核心: 严格模拟 CUDA `__float2bfloat16` 的 Round-to-nearest-even (向偶数舍入) 行为。
    """
    (b,) = struct.unpack("<I", struct.pack("<f", x))
    if math.isnan(x):
        return 0x7FC0
    rounding_bias = ((b >> 16) & 1) + 0x7FFF
    return ((b + rounding_bias) >> 16) & 0xFFFF


def encode_bf16(values) -> bytes:
    return b"".join(struct.pack("<H", _f32_to_bf16_bits(v)) for v in values)


def decode_native(raw: bytes, dtype: str) -> array:
    if dtype == "fp16":
        return decode_fp16(raw)
    if dtype == "bf16":
        return decode_bf16(raw)
    raise ValueError(f"unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# 官方 fast_hadamard_transform 库封装
# ---------------------------------------------------------------------------

def library_available() -> bool:
    """检查官方库及依赖是否已正确安装。"""
    try:
        import torch  # noqa: F401
        import fast_hadamard_transform  # noqa: F401
        return True
    except ImportError:
        return False


def _require_library():
    if not library_available():
        raise RuntimeError(
            "官方库参考不可用：需要 torch + fast_hadamard_transform。\n"
            "请按 README '依赖安装' 一节搭建 ~/hadamard_env，\n"
            "并用 ~/hadamard_env/bin/python 运行本脚本。")


def library_hadamard_from_native(raw_in: bytes, dtype: str, head_dim: int,
                                 normalize: bool):
    """调用官方库计算标准参考输出。

    参数:
      raw_in:    小端序低精度输入字节流 (长度 = total_tokens * head_dim * 2)
      dtype:     "fp16" 或 "bf16"
      head_dim:  变换维度 (必须为 2 的幂)
      normalize: 是否归一化 (True: scale=1/sqrt(d), False: scale=1.0)

    返回:
      展平后的 CPU 端 float32 列表，用于后续误差计算。
    """
    _require_library()
    import torch
    from fast_hadamard_transform import hadamard_transform

    if not torch.cuda.is_available():
        raise RuntimeError("官方库参考需要可用的 CUDA 设备")
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16

    x = torch.frombuffer(bytearray(raw_in), dtype=tdt).reshape(-1, head_dim).cuda()

    scale = 1.0 / math.sqrt(head_dim) if normalize else 1.0
    ref = hadamard_transform(x, scale)

    # 转回 FP32 并展平，移回 CPU 以便与 Python 原生数据对比
    return ref.float().reshape(-1).cpu()


def reference_from_native(raw_in: bytes, dtype: str, head_dim: int,
                          normalize: bool):
    """统一对外接口：返回 (参考值列表, 数据来源标识)。"""
    t = library_hadamard_from_native(raw_in, dtype, head_dim, normalize)
    return t.tolist(), "fast_hadamard_transform"


# ---------------------------------------------------------------------------
# 自检：只依赖官方库自身的代数性质
# ---------------------------------------------------------------------------

def _selftest():
    _require_library()
    import torch
    from fast_hadamard_transform import hadamard_transform

    d = 128 # 测试维度

    # 测试 1: 脉冲响应 (Impulse Response)
    # 原理: H · e_0 等价于提取 Hadamard 矩阵的第 0 列。
    # 在 Sylvester 自然序下，第 0 列的所有元素均为 +1。
    e0 = torch.zeros(1, d, dtype=torch.float16, device="cuda")
    e0[0, 0] = 1.0
    y = hadamard_transform(e0).float()
    assert torch.all(y == 1.0), f"impulse e0 应映射到全 1 列，实际 {y[0, :8]}"

    # 测试 2: 正交对合性 (Involutory Property)
    # 原理: 归一化的 Hadamard 矩阵 (H/√d) 是正交且对合的，即 (H/√d)² = I。
    # 连续进行两次归一化变换，必须完美还原原始输入。
    torch.manual_seed(0)
    x = torch.randn(7, d, dtype=torch.float16, device="cuda")
    s = 1.0 / math.sqrt(d)
    z = hadamard_transform(hadamard_transform(x, s), s)
    err = (z.float() - x.float()).abs().max().item()
    assert err < 2e-2, f"归一化两次应还原输入，最大偏差 {err}"

    # 测试 3: 能量守恒 (Parseval's Theorem / 保范数)
    # 原理: 正交矩阵变换不改变向量的 L2 范数平方 (能量)。
    e_in = x.float().pow(2).sum().item()
    e_out = hadamard_transform(x, s).float().pow(2).sum().item()
    assert abs(e_in - e_out) / e_in < 1e-3, f"能量不守恒: {e_in} vs {e_out}"
    print("official-library reference selftest: OK "
          "(impulse 全 1 列 / 归一化对合还原 / 保范数)")


if __name__ == "__main__":
    _selftest()
