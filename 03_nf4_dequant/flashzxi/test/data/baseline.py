import struct
import torch
import bitsandbytes.functional as F
import time

def _dequant_bnb(qweight: torch.Tensor, qs):
    """
    兼容不同 bnb 版本的反量化入口：
    优先用 dequantize_4bit；没有的话再退到 dequantize_blockwise。
    """
    if hasattr(F, "dequantize_4bit"):
        # 新版常见：直接传 quant_state
        return F.dequantize_4bit(qweight, quant_state=qs)
    if hasattr(F, "dequantize_blockwise"):
        # 老版可能需要 absmax/code 等；但如果传 quant_state 通常也能工作
        return F.dequantize_blockwise(qweight, quant_state=qs)
    raise RuntimeError("当前 bitsandbytes.functional 里找不到 dequantize_4bit / dequantize_blockwise")

def save_nf4_tagged_binary(path: str, W: torch.Tensor, blocksize: int = 64):
    """
    写 w_nf4.bin：
      [header]\n
      num_rows:<int64 binary>\n
      num_cols:<int64 binary>\n
      blocksize:<int32 binary>\n\n
      [data]\n
      packed_weights:<uint8 blob>\n
      absmax_q:<uint8 blob>\n
      absmax2:<float16 blob>\n
      code2:<float16[256] blob>\n
      offset:<float32 binary>\n
    """
    assert W.ndim == 2 and W.is_cuda

    num_rows, num_cols = map(int, W.shape)
    num_elements = num_rows * num_cols
    num_blocks = (num_elements + blocksize - 1) // blocksize

    qweight, qs = F.quantize_4bit(
        W,
        blocksize=blocksize,
        quant_type="nf4",
        compress_statistics=True,
        quant_storage=torch.uint8,
    )
    if not getattr(qs, "nested", False) or qs.state2 is None:
        raise RuntimeError("需要 compress_statistics=True 才会有 absmax_q/absmax2/code2/offset")

    packed = qweight.detach().contiguous().view(torch.uint8).cpu()
    packed_len = (num_elements + 1) // 2
    if packed.numel() != packed_len:
        raise RuntimeError(f"packed_weights len mismatch: got={packed.numel()} expected={packed_len}")

    absmax_q = qs.absmax.detach().contiguous().view(torch.uint8).cpu()
    if absmax_q.numel() != num_blocks:
        raise RuntimeError(f"absmax_q len mismatch: got={absmax_q.numel()} expected={num_blocks}")

    absmax2 = qs.state2.absmax.detach().contiguous().cpu().to(torch.float16)
    code2 = qs.state2.code.detach().contiguous().cpu().to(torch.float16)
    if code2.numel() != 256:
        raise RuntimeError(f"code2 len mismatch: got={code2.numel()} expected=256")

    offset = float(qs.offset) if qs.offset is not None else 0.0

    with open(path, "wb") as f:
        f.write(b"[header]\n")
        f.write(b"num_rows: ")
        f.write(struct.pack("<q", num_rows))
        f.write(b"\nnum_cols: ")
        f.write(struct.pack("<q", num_cols))
        f.write(b"\nblocksize: ")
        f.write(struct.pack("<i", int(blocksize)))
        f.write(b"\n\n")

        f.write(b"[data]\n")
        f.write(b"packed_weights: ")
        f.write(packed.numpy().tobytes(order="C"))
        f.write(b"\nabsmax_q: ")
        f.write(absmax_q.numpy().tobytes(order="C"))
        f.write(b"\nabsmax2: ")
        f.write(absmax2.numpy().tobytes(order="C"))
        f.write(b"\ncode2: ")
        f.write(code2.numpy().tobytes(order="C"))
        f.write(b"\noffset: ")
        f.write(struct.pack("<f", offset))
        f.write(b"\n")

    return qweight, qs, (num_rows, num_cols)

def save_dequant_result(path: str, deq: torch.Tensor, shape, out_dtype=torch.float16):
    """
    写 w_dequant.bin：
      [dequant]\n
      num_rows:<int64>\n
      num_cols:<int64>\n
      dtype:<1 byte tag>\n
      data:<raw blob>\n

    dtype tag: 1 = fp16, 2 = fp32
    """
    num_rows, num_cols = shape
    deq2d = deq.reshape(num_rows, num_cols).detach()

    if out_dtype == torch.float16:
        tag = 1
        host = deq2d.to(torch.float16).contiguous().cpu()
    elif out_dtype == torch.float32:
        tag = 2
        host = deq2d.to(torch.float32).contiguous().cpu()
    else:
        raise ValueError("out_dtype 只支持 torch.float16 或 torch.float32")

    with open(path, "wb") as f:
        f.write(host.numpy().tobytes(order="C"))

    # with open(path + ".txt", "w", encoding="utf-8") as f:
    #     f.write("[dequant]\n")
    #     f.write(f"num_rows: {int(num_rows)}\n")
    #     f.write(f"num_cols: {int(num_cols)}\n")
    #     f.write(f"dtype: {int(tag)}\n")
    #     f.write("data:\n")
    #
    #     # 逐行写，空格分隔
    #     # 可以按需要改格式，比如 "{:.6f}"
    #     for i in range(num_rows):
    #         row = host[i].tolist()
    #         f.write(" ".join(f"{v:.6f}" for v in row))
    #         f.write("\n")
    #         # print(" ".join(f"{v:.6f}" for v in row))

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.manual_seed(1234)              # CPU RNG
    torch.cuda.manual_seed_all(1234)     # 所有 GPU RNG（单卡也可用）
    row = 10000
    col = 10000
    W = torch.randn(row, col, device="cuda", dtype=torch.float16)
    file_prefix = f"nf4_{row}x{col}_fp16"
    qweight, qs, shape = save_nf4_tagged_binary(file_prefix + ".bin", W, blocksize=64)

    start = time.perf_counter()
    deq = _dequant_bnb(qweight, qs)  # bnb 反量化
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"dequantize_4bit执行时间: {elapsed_ms:.3f} ms")
    save_dequant_result(file_prefix + "_w_dequant.bin", deq, shape, out_dtype=torch.float16)
