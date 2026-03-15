import torch
import bitsandbytes.functional as F


def main():
    assert torch.cuda.is_available(), "CUDA"

    device = "cuda"

    # =========================
    # 测试规模
    # =========================
    rows = 16384
    cols = 16384
    blocksize = 64
    repeat = 10

    print("=== 4bit Dequant Bandwidth Test ===")

    # =========================
    # 构造数据
    # =========================
    x = torch.randn(
        rows,
        cols,
        device=device,
        dtype=torch.float16
    )

    # 4bit 量化
    q_weight, state = F.quantize_4bit(
        x,
        blocksize=blocksize,
        compress_statistics=True
    )

    numel = q_weight.numel()

    # =========================
    # 预热
    # =========================
    for _ in range(3):
        F.dequantize_4bit(q_weight, state)

    torch.cuda.synchronize()

    # =========================
    # CUDA 计时
    # =========================
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()

    for _ in range(repeat):
        y = F.dequantize_4bit(q_weight, state)

    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)

    # =========================
    # 带宽估算
    # =========================

    # 读取：
    # 1. 4bit weight: 0.5 byte
    # 2. absmax (fp16): 2 byte
    #
    # 写入：
    # 3. fp16 output: 2 byte

    bytes_read = numel * 0.5 + numel * 2
    bytes_write = numel * 2

    total_bytes = (bytes_read + bytes_write) * repeat

    seconds = elapsed_ms / 1000.0

    gbps = total_bytes / seconds / 1e9

    # =========================
    # 输出
    # =========================
    print(f"Matrix: {rows} x {cols}")
    print(f"Repeat: {repeat}")
    print(f"Time: {elapsed_ms:.3f} ms")
    print(f"Bandwidth: {gbps:.2f} GB/s")


if __name__ == "__main__":
    main()