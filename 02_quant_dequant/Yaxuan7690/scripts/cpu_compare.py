#!/usr/bin/env python3
# =============================================================================
#  cpu_compare.py -- 在较大矩阵上做 GPU 与 CPU 同级别实现的稳定对比
#
#  小矩阵(512x512)下 GPU kernel 只有几微秒, 加速比被 kernel 启动开销主导, 单次测量
#  抖动很大(几十倍量级). 这里改用 2048x2048 = 419 万元素, 让 GPU 端进入稳态,
#  同时 CPU 参考实现仍可在百毫秒量级跑完.
#
#  用法: python scripts/cpu_compare.py [--n 2048] [--repeat 20]
#  产出: data/cpu_compare.csv
# =============================================================================
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_experiments import gen_tensor, write_params, parse_log, FORMATS  # noqa: E402

# 覆盖两种格式 x 两种缩放 x 两种舍入 的代表性组合
CONFIGS = [
    ("mxfp8", "e4m3", 32, "block", "fp16", "nearest"),
    ("mxfp8", "e4m3", 32, "block", "fp16", "stochastic"),
    ("mxfp8", "e5m2", 32, "block", "fp16", "nearest"),
    ("mxfp4", "e2m1", 32, "block", "fp16", "nearest"),
    ("nvfp4", None, 16, "block", "fp16", "nearest"),
    ("nvfp4", None, 16, "block", "fp16", "stochastic"),
    ("nvfp4", None, 16, "tensor", "fp16", "nearest"),
    ("nvfp4", None, 16, "tensor", "fp32", "nearest"),
]
OUTPUT_TYPES = ("fp16", "bf16", "fp32")
ROUNDINGS = ("nearest", "stochastic")


def cfg_name(fmt, elem, bs, sm, ot, rd):
    return "%s-%s_%s_%s_%s" % (fmt, elem or "e2m1", sm, ot, rd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("build", "lpq_quant.exe"))
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    tdir, pdir, odir = (os.path.join(ROOT, "data", d) for d in ("tensors", "params", "out"))
    for d in (tdir, pdir, odir):
        os.makedirs(d, exist_ok=True)
    gpu = "RTX 5060 Ti (sm_120, CUDA 12.9)"

    tname = "normal_%d" % args.n
    gen_tensor(tdir, tname, args.n, args.n, "normal", "fp32", 999)

    rows = []
    for fmt, elem, bs, sm, ot, rd in CONFIGS:
        cname = cfg_name(fmt, elem, bs, sm, ot, rd)
        pfile = os.path.join(pdir, "cpu_%s.params" % cname)
        write_params(pfile, fmt, elem, bs, sm, ot, rd, gpu)
        pref = "cpu_%s__%s" % (tname, cname)
        cmd = [os.path.abspath(os.path.join(ROOT, args.exe)),
               "--input", "data/tensors/%s.tensor" % tname,
               "--params", "data/params/cpu_%s.params" % cname,
               "--out-dir", "data/out", "--out-prefix", pref,
               "--repeat", str(args.repeat), "--warmup", str(args.warmup),
               "--cpu-check", "--quiet"]
        subprocess.run(cmd, check=True, cwd=ROOT)
        lg = parse_log(os.path.join(odir, pref + "_log.txt"))
        # 只保留日志, 删掉体积大的二进制产物
        for suf in ("_lowp.bin", "_dequant.bin"):
            p = os.path.join(odir, pref + suf)
            if os.path.exists(p):
                os.remove(p)
        rows.append({
            "numel": int(lg["input.numel"]), "config": cname,
            "gpu_quantize_ms": lg["performance.quantize_total_ms"],
            "gpu_dequantize_ms": lg["performance.dequantize_kernel_ms"],
            "cpu_quantize_ms": lg["accuracy_check_cpu_reference.cpu_quantize_ms"],
            "cpu_dequantize_ms": lg["accuracy_check_cpu_reference.cpu_dequantize_ms"],
            "speedup_quantize": lg["accuracy_check_cpu_reference.speedup_quantize_gpu_vs_cpu"],
            "speedup_dequantize":
                lg["accuracy_check_cpu_reference.speedup_dequantize_gpu_vs_cpu"],
            "packed_bits_identical": lg["accuracy_check_cpu_reference.packed_bits_identical"],
            "dequant_bit_mismatch_count":
                lg["accuracy_check_cpu_reference.dequant_bit_mismatch_count"],
        })
        print("  %-30s q: %.1fx   dq: %.1fx   bits_identical=%s mism=%s"
              % (cname, float(rows[-1]["speedup_quantize"]), float(rows[-1]["speedup_dequantize"]),
                 rows[-1]["packed_bits_identical"], rows[-1]["dequant_bit_mismatch_count"]))

    out = os.path.join(ROOT, "data", "cpu_compare.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()