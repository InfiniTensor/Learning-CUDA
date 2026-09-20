#!/usr/bin/env python3
# =============================================================================
#  analyze.py -- 把 data/*.csv 汇总成报告用的 Markdown 表格
#
#  用法: python scripts/analyze.py > docs/measurements.md
# =============================================================================
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_experiments import SCALE_RULES  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # 固定 LF, 别让 Windows 输出 CRLF


def load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(io.open(p, encoding="utf-8")))


def f(x, n=6):
    try:
        return ("%%.%dg" % n) % float(x)
    except (TypeError, ValueError):
        return "-"


def table(header, rows, align=None):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(align or (["---"] * len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    rows = load("summary.csv")
    bench = load("benchmark.csv")
    bench_old = load("benchmark_before_fusion.csv")

    # 主矩阵的四个格式; 4/6 对照用的配置另有 format 名, 不进这三张表
    order = {"mxfp8-e4m3": 0, "mxfp8-e5m2": 1, "nvfp4": 2, "mxfp4": 3}

    # ---------- 1. 精度: 三种分布 x 四种格式 (512x512, RNE, fp16) ----------
    print("## 表 1  误差指标(512x512, rounding=nearest, output_type=fp16)\n")
    sel = [r for r in rows if r["tensor"].endswith("_512")
           and r["output_type"] == "fp16" and r["rounding"] == "nearest"
           and r["format"] in order]
    t = []
    for r in sorted(sel, key=lambda r: (r["dist"], order[r["format"]], r["scale_mode"])):
        t.append([r["dist"], r["format"], r["scale_mode"], f(r["compression_ratio"], 5),
                  f(r["max_abs_error"], 4), f(r["mae"], 4), f(r["mse"], 4),
                  f(r["relative_mse"], 4)])
    print(table(["分布", "格式", "缩放", "压缩率", "最大绝对误差", "MAE", "MSE", "相对 MSE"], t))
    print()

    # ---------- 2. 随机舍入 vs 就近舍入 ----------
    print("\n## 表 2  随机舍入(SR)与就近舍入(RNE)的误差与偏置(512x512, fp16)\n")
    t = []
    for r in sorted(sel, key=lambda r: (r["dist"], order[r["format"]], r["scale_mode"])):
        m = [q for q in rows if q["tensor"] == r["tensor"] and q["config"] ==
             r["config"].replace("_nearest", "_stochastic")]
        t.append([r["dist"], r["format"], r["scale_mode"],
                  f(r["mae"], 4), f(r["mse"], 4),
                  f(m[0]["mae"], 4) if m else "-", f(m[0]["mse"], 4) if m else "-"])
    print(table(["分布", "格式", "缩放", "MAE(RNE)", "MSE(RNE)", "MAE(SR)", "MSE(SR)"], t))
    print()

    # ---------- 3. 压缩率 ----------
    print("\n## 表 3  压缩率与每元素位宽\n")
    t = []
    for fmt in ("mxfp8-e4m3", "mxfp8-e5m2", "nvfp4", "mxfp4"):
        for sm in ("tensor", "block"):
            c = [r for r in rows if r["format"] == fmt and r["scale_mode"] == sm]
            if not c:
                continue
            t.append([fmt, sm, c[0]["block_size"], f(c[0]["packed_bytes"], 6),
                      f(c[0]["scale_bytes"], 6), f(c[0]["bits_per_element"], 5),
                      f(c[0]["compression_ratio"], 5)])
    print(table(["格式", "缩放", "block_size", "packed Bytes", "scale Bytes",
                 "bit/元素", "压缩率"], t))
    print()

    # ---------- 4. 性能随规模变化 ----------
    print("\n## 表 4  性能随矩阵规模的变化(正态分布, output_type=fp16)\n")
    for cfg in ("mxfp8-e4m3_block_fp16_nearest", "mxfp8-e5m2_block_fp16_nearest",
                "mxfp4-e2m1_block_fp16_nearest",
                "nvfp4-e2m1_block_fp16_nearest", "nvfp4-e2m1_tensor_fp16_nearest"):
        t = []
        for r in sorted([x for x in bench if x["config"] == cfg], key=lambda x: int(x["n"])):
            t.append([int(r["n"]), int(r["numel"]), f(r["quantize_total_ms"], 4),
                      f(r["dequantize_ms"], 4), f(r["bw_quant_GBps"], 4),
                      f(r["bw_dequant_GBps"], 4)])
        if t:
            print("**%s**\n" % cfg)
            print(table(["n", "元素数", "量化 ms", "反量化 ms", "量化 GB/s", "反量化 GB/s"], t))
            print()

    # ---------- 5. 融合优化前后 ----------
    print("\n## 表 5  融合 kernel 优化前后(正态分布, block 模式, fp16)\n")
    t = []
    for cfg in ("mxfp8-e4m3_block_fp16_nearest", "mxfp8-e5m2_block_fp16_nearest"):
        for r in sorted([x for x in bench if x["config"] == cfg], key=lambda x: int(x["n"])):
            o = [y for y in bench_old if y["config"] == cfg and y["n"] == r["n"]]
            if not o:
                continue
            a, b = float(o[0]["quantize_total_ms"]), float(r["quantize_total_ms"])
            t.append([int(r["n"]), f(a, 4), f(b, 4), "%.2fx" % (a / b) if b > 0 else "-",
                      f(o[0]["bw_quant_GBps"], 4), f(r["bw_quant_GBps"], 4)])
    print(table(["n", "量化 ms(三阶段)", "量化 ms(融合)", "加速比",
                 "量化 GB/s(前)", "量化 GB/s(后)"], t))
    print()

    # ---------- 6. GPU vs CPU ----------
    print("\n## 表 6  GPU 与 CPU 同级别实现的耗时对比(512x512)\n")
    t = []
    for r in sorted([x for x in rows if x["tensor"].endswith("_512")
                     and x["output_type"] == "fp16" and x["rounding"] == "nearest"
                     and x["format"] in order],
                    key=lambda r: (r["dist"], order[r["format"]], r["scale_mode"])):
        if not r["cpu_quantize_ms"]:
            continue
        t.append([r["dist"], r["config"], f(r["cpu_quantize_ms"], 5),
                  f(r["quantize_total_ms"], 4), f(r["speedup_quantize"], 4),
                  f(r["cpu_dequantize_ms"], 5), f(r["dequantize_ms"], 4),
                  f(r["speedup_dequantize"], 4)])
    print(table(["分布", "配置", "CPU 量化 ms", "GPU 量化 ms", "加速比",
                 "CPU 反量化 ms", "GPU 反量化 ms", "加速比"], t))
    print()

    # ---------- 7. 4/6 自适应块缩放 vs static_6 基线 ----------
    print("\n## 表 7  4/6 自适应块缩放逐规则对照(512x512, nvfp4, block=16, fp16, RNE)\n")
    t = []
    for dist in ("random", "normal", "outlier"):
        base = [r for r in rows if r["dist"] == dist
                and r["config"] == "nvfp4_block_fp16_nearest"]
        if not base:
            continue
        base = base[0]
        cases = [("static_6 (基线)", "nvfp4_block_fp16_nearest")]
        cases += [(rule, "%s_block_fp16_nearest" % nm) for nm, rule in SCALE_RULES]
        for label, cfg in cases:
            hit = [r for r in rows if r["dist"] == dist and r["config"] == cfg
                   and r["output_type"] == "fp16" and r["rounding"] == "nearest"]
            if not hit:
                continue
            r = hit[0]
            d_mae = 100.0 * (float(r["mae"]) / float(base["mae"]) - 1.0)
            d_mse = 100.0 * (float(r["mse"]) / float(base["mse"]) - 1.0)
            d_max = 100.0 * (float(r["max_abs_error"]) / float(base["max_abs_error"]) - 1.0)
            t.append([dist, label, f(r["mae"], 4), f(r["mse"], 4), f(r["relative_mse"], 4),
                      "%+.2f%%" % d_mae, "%+.2f%%" % d_mse, "%+.2f%%" % d_max])
    print(table(["分布", "块缩放规则", "MAE", "MSE", "相对 MSE",
                 "MAE vs 基线", "MSE vs 基线", "max|err| vs 基线"], t))
    print()


if __name__ == "__main__":
    main()