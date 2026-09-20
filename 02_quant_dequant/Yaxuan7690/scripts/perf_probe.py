#!/usr/bin/env python3
# =============================================================================
#  perf_probe.py -- 优化迭代用的快速性能台架
#
#  固定用一个大尺寸方阵 (默认 8192, 即 6710 万元素), 跑 MXFP8 与 NVFP4 两条链路,
#  把日志里的每个阶段 kernel 时间换算成有效带宽并与显存峰值对比, 打印成一张表.
#  目的是让"改一个 kernel -> 看它动了多少"这个循环尽可能短.
#
#  用法:
#      python scripts/perf_probe.py                 # 默认 8192, repeat 20
#      python scripts/perf_probe.py --n 4096
#      python scripts/perf_probe.py --save base     # 存快照到 data/perf_<name>.csv
#      python scripts/perf_probe.py --compare base  # 与快照对比
# =============================================================================
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_experiments import gen_tensor, write_params, parse_log  # noqa: E402

PEAK_GBPS = 448.0  # RTX 5060 Ti, 128-bit GDDR7 @ 28 Gbps

# (显示名, format, elem, block_size, scale_mode, output_type, rounding)
CASES = [
    ("mxfp8-e4m3/block", "mxfp8", "e4m3", 32, "block", "fp16", "nearest"),
    ("nvfp4/block", "nvfp4", None, 16, "block", "fp16", "nearest"),
    ("mxfp8-e4m3/block-sr", "mxfp8", "e4m3", 32, "block", "fp16", "stochastic"),
    ("nvfp4/block-sr", "nvfp4", None, 16, "block", "fp16", "stochastic"),
]

STAGES = [
    ("amax", "amax_kernel_ms"),
    ("scale", "scale_kernel_ms"),
    ("quant", "quantize_kernel_ms"),
    ("dequant", "dequantize_kernel_ms"),
    ("metrics", "metrics_kernel_ms"),
]


def bytes_moved(n, fmt, sm):
    """各阶段搬运的字节数 (读+写)."""
    x = n * 4                      # 输入 fp32
    if fmt == "mxfp8":
        packed, scale = n, n // 32
    else:
        packed, scale = n // 2, n // 16
    out16 = n * 2                  # fp16 反量化输出
    return {
        "amax": x,
        "scale": 0,
        "quant": x + packed + scale,
        "dequant": packed + scale + out16,
        "metrics": x + out16,  # 原始输入(fp32) + 反量化输出(fp16)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("build", "lpq_quant.exe"))
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    tdir, pdir, odir = (os.path.join(ROOT, "data", d) for d in ("tensors", "params", "out"))
    for d in (tdir, pdir, odir):
        os.makedirs(d, exist_ok=True)
    tname = "perf_%d" % args.n
    gen_tensor(tdir, tname, args.n, args.n, "normal", "fp32", 999)

    n = args.n * args.n
    results = {}
    print("n = %d  (%d 元素, 峰值 %.0f GB/s)\n" % (args.n, n, PEAK_GBPS))
    for label, fmt, elem, bs, sm, ot, rd in CASES:
        pfile = os.path.join(pdir, "perf_%s.params" % label.replace("/", "_"))
        write_params(pfile, fmt, elem, bs, sm, ot, rd, "RTX 5060 Ti (sm_120, CUDA 12.9)")
        pref = "perf_%d__%s" % (args.n, label.replace("/", "_"))
        subprocess.run(
            [os.path.abspath(os.path.join(ROOT, args.exe)),
             "--input", "data/tensors/%s.tensor" % tname,
             "--params", "data/params/perf_%s.params" % label.replace("/", "_"),
             "--out-dir", "data/out", "--out-prefix", pref,
             "--repeat", str(args.repeat), "--warmup", str(args.warmup), "--quiet"],
            check=True, cwd=ROOT)
        lg = parse_log(os.path.join(odir, pref + "_log.txt"))

        moved = bytes_moved(n, fmt, sm)
        print("== %s ==" % label)
        print("  %-8s %10s %10s %9s" % ("stage", "ms", "GB/s", "% peak"))
        tot = 0.0
        row = {}
        for name, key in STAGES:
            ms = float(lg["performance." + key])
            tot += ms
            b = moved[name]
            bw = (b / 1e9) / (ms / 1e3) if ms > 0 else 0.0
            row[name] = ms
            print("  %-8s %10.4f %10.1f %8.1f%%" % (name, ms, bw, 100.0 * bw / PEAK_GBPS))
        print("  %-8s %10.4f" % ("TOTAL", tot))
        row["total"] = tot
        results[label] = row
        print()

    if args.save:
        out = os.path.join(ROOT, "data", "perf_%s.csv" % args.save)
        keys = list(next(iter(results.values())).keys())
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case"] + keys)
            for label, row in results.items():
                w.writerow([label] + [row[k] for k in keys])
        print("saved", out)

    if args.compare:
        old = {}
        with open(os.path.join(ROOT, "data", "perf_%s.csv" % args.compare),
                  encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                old[r["case"]] = r
        print("=== 与 %s 对比 ===" % args.compare)
        for label, row in results.items():
            o = old.get(label)
            if not o:
                continue
            for k in row:
                a, b = float(o[k]), row[k]
                if a > 0:
                    print("  %-18s %-8s %8.4f -> %8.4f ms  (x%.2f)"
                          % (label, k, a, b, a / b if b > 0 else 0))


if __name__ == "__main__":
    main()
