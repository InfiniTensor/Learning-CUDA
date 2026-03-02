import argparse
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

NF4_LUT = np.array(
    [
        -1.00000000,
        -0.69619280,
        -0.52507305,
        -0.39491710,
        -0.28444138,
        -0.18477343,
        -0.09105003,
        0.00000000,
        0.07958030,
        0.16093020,
        0.24611230,
        0.33791524,
        0.44070983,
        0.56261700,
        0.72295684,
        1.00000000,
    ],
    dtype=np.float32,
)


def cpu_dequant(packed, absmax_q, absmax2, code2, offset, total_elements, blocksize):
    out = np.empty(total_elements, dtype=np.float16)
    for byte_idx, val in enumerate(packed):
        base = byte_idx * 2
        if base >= total_elements:
            break

        block_idx = base // blocksize
        group_idx = block_idx // 256
        scale = np.float32(code2[absmax_q[block_idx]]) * np.float32(absmax2[group_idx]) + np.float32(offset)

        out[base] = np.float16(NF4_LUT[(val >> 4) & 0xF] * scale)
        if base + 1 < total_elements:
            out[base + 1] = np.float16(NF4_LUT[val & 0xF] * scale)
    return out


def gen_case(rows, cols, blocksize, seed=0):
    rng = np.random.default_rng(seed)
    total_elements = rows * cols
    num_packed = (total_elements + 1) // 2
    num_blocks = (total_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + 255) // 256

    packed = rng.integers(0, 256, size=num_packed, dtype=np.uint8)
    absmax_q = rng.integers(0, 256, size=num_blocks, dtype=np.uint8)
    absmax2 = (rng.random(num_groups, dtype=np.float32) * 2.0).astype(np.float16)
    code2 = np.linspace(0.01, 1.5, 256, dtype=np.float32).astype(np.float16)
    offset = np.float32(-0.02)
    return packed, absmax_q, absmax2, code2, offset


def write_weight_bin(path, rows, cols, blocksize, packed, absmax_q, absmax2, code2, offset):
    with open(path, "wb") as f:
        f.write(struct.pack("q", rows))
        f.write(struct.pack("q", cols))
        f.write(struct.pack("i", blocksize))
        f.write(packed.tobytes())
        f.write(absmax_q.tobytes())
        f.write(absmax2.tobytes())
        f.write(code2.tobytes())
        f.write(struct.pack("f", float(offset)))


def maybe_build(mx_dir: Path):
    exe = mx_dir / "nf4_dequant_mx"
    if exe.exists():
        return exe

    compiler = shutil.which("mxcc") or shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("未找到 mxcc/nvcc，无法编译 nf4_dequant_mx。")

    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        "-o",
        "nf4_dequant_mx",
        "nf4_dequant_mx.cu",
    ]
    if compiler.endswith("nvcc"):
        cmd += ["-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart"]

    subprocess.run(cmd, cwd=mx_dir, check=True)
    return exe


def main():
    parser = argparse.ArgumentParser(description="NF4 沐曦实现功能测试")
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--cols", type=int, default=253)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=2e-2)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    mx_dir = Path(__file__).resolve().parent
    project_dir = mx_dir.parent
    weight_dir = project_dir / "weight_data"
    mx_results = project_dir / "mx_results"

    weight_dir.mkdir(parents=True, exist_ok=True)
    mx_results.mkdir(parents=True, exist_ok=True)

    packed, absmax_q, absmax2, code2, offset = gen_case(
        args.rows, args.cols, args.blocksize, seed=args.seed
    )
    total_elements = args.rows * args.cols

    case_name = f"weight_{args.rows}x{args.cols}_bs{args.blocksize}_mx_test.bin"
    weight_file = weight_dir / case_name
    write_weight_bin(
        weight_file,
        args.rows,
        args.cols,
        args.blocksize,
        packed,
        absmax_q,
        absmax2,
        code2,
        offset,
    )

    if not args.no_build:
        maybe_build(mx_dir)

    exe = mx_dir / "nf4_dequant_mx"
    if not exe.exists():
        raise RuntimeError("找不到 nf4_dequant_mx，可先执行 run_all.sh 或手动编译。")

    subprocess.run([str(exe), str(weight_file)], cwd=mx_dir, check=True)

    out_file = mx_results / f"dequant_{args.rows}x{args.cols}_bs{args.blocksize}.fp16"
    if not out_file.exists():
        raise RuntimeError(f"未生成输出文件: {out_file}")

    gpu_out = np.fromfile(out_file, dtype=np.float16)
    ref_out = cpu_dequant(
        packed, absmax_q, absmax2, code2, offset, total_elements, args.blocksize
    )

    mae = float(np.mean(np.abs(gpu_out.astype(np.float32) - ref_out.astype(np.float32))))
    max_diff = float(np.max(np.abs(gpu_out.astype(np.float32) - ref_out.astype(np.float32))))

    print(f"MAE: {mae:.8f}")
    print(f"MaxDiff: {max_diff:.8f}")

    if max_diff > args.tol:
        raise SystemExit(f"测试失败: MaxDiff={max_diff:.8f} > tol={args.tol}")

    print("测试通过")


if __name__ == "__main__":
    main()
