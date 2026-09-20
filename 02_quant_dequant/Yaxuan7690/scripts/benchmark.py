#!/usr/bin/env python3
# =============================================================================
#  benchmark.py -- 规模扫描: 固定配置, 改变矩阵规模, 观察有效带宽随规模的变化
#
#  说明:
#    * 小规模(<= 512)时 kernel 时间被启动开销/尾部效应主导, 有效带宽无法反映真实
#      峰值; 因此必须给出「规模 -> 带宽」曲线, 才能说明实现是否达到显存带宽上限.
#    * 这里扫 [KB] -> [千万元素], 覆盖 launch-bound 到 bandwidth-bound 两个区间.
#
#  用法:  python scripts/benchmark.py [--repeat 50] [--warmup 10]
#  产出:  data/benchmark.csv  +  stdout 表格
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

SIZES = [256, 512, 1024, 2048, 4096, 8192]
CONFIGS = [
    ("mxfp8", "e4m3", 32, "block", "fp16", "nearest"),
    ("mxfp8", "e4m3", 32, "block", "fp16", "stochastic"),
    ("mxfp8", "e5m2", 32, "block", "fp16", "nearest"),
    ("mxfp4", "e2m1", 32, "block", "fp16", "nearest"),
    ("nvfp4", None, 16, "block", "fp16", "nearest"),
    ("nvfp4", None, 16, "block", "fp16", "stochastic"),
    ("nvfp4", None, 16, "tensor", "fp16", "nearest"),
]


def cfg_name(fmt, elem, bs, sm, ot, rd):
    return "%s-%s_%s_%s_%s" % (fmt, elem or "e2m1", sm, ot, rd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("build", "lpq_quant.exe"))
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    tdir, pdir, odir = (os.path.join(ROOT, "data", d) for d in ("tensors", "params", "out"))
    for d in (tdir, pdir, odir):
        os.makedirs(d, exist_ok=True)
    gpu = "RTX 5060 Ti (sm_120, CUDA 12.9)"

    rows = []
    for n in SIZES:
        tname = "normal_%d" % n
        gen_tensor(tdir, tname, n, n, "normal", "fp32", 999)
        for fmt, elem, bs, sm, ot, rd in CONFIGS:
            cname = cfg_name(fmt, elem, bs, sm, ot, rd)
            pfile = os.path.join(pdir, "bench_%s.params" % cname)
            write_params(pfile, fmt, elem, bs, sm, ot, rd, gpu)
            pref = "bench_%s__%s" % (tname, cname)
            cmd = [os.path.abspath(os.path.join(ROOT, args.exe)),
                   "--input", "data/tensors/%s.tensor" % tname,
                   "--params", "data/params/bench_%s.params" % cname,
                   "--out-dir", "data/out", "--out-prefix", pref,
                   "--repeat", str(args.repeat), "--warmup", str(args.warmup), "--quiet"]
            subprocess.run(cmd, check=True, cwd=ROOT)
            lg = parse_log(os.path.join(odir, pref + "_log.txt"))
            rows.append({
                "n": n, "numel": int(lg["input.numel"]), "config": cname,
                "compression_ratio": lg["lowp.compression_ratio"].split()[0],
                "quantize_total_ms": lg["performance.quantize_total_ms"],
                "dequantize_ms": lg["performance.dequantize_kernel_ms"],
                "input_passes": lg["performance.quantize_input_passes"],
                "bw_quant_GBps": lg["performance.effective_bandwidth_quantize_GBps"],
                "dram_bw_quant_GBps": lg["performance.dram_bandwidth_quantize_GBps"],
                "bw_dequant_GBps": lg["performance.effective_bandwidth_dequantize_GBps"],
                "max_abs_error": lg["error.max_abs_error"],
                "relative_mse": lg["error.relative_mse"],
            })
            print("  n=%5d %-28s q=%8s ms  dq=%8s ms  bw_dq=%8s GB/s"
                  % (n, cname, rows[-1]["quantize_total_ms"], rows[-1]["dequantize_ms"],
                     rows[-1]["bw_dequant_GBps"]))

    out = os.path.join(ROOT, "data", "benchmark.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()