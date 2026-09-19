#!/usr/bin/env python3
"""Turn the MUSA (Moore Threads / MTT S4000) experiment artefacts into tables.

The CUDA tables come from `tools/analyze.py` over `data/*.csv`.  This is the same
job for the second platform: it reads `data/musa/*` and writes a separate
document, because the two platforms' numbers must never be mixed in one table --
the runtime, the roof and the launch cost all differ (see the report).

Usage:
    python tools/analyze_musa.py [--data data/musa] [--out docs/结果表格_MUSA.md]

Everything is optional: sections whose inputs are missing are skipped, so the
tool can be run while a sweep is still in flight.
"""

import argparse
import csv
import os
import re

DTYPES = ("f16", "bf16")
DIMS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def num(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def cell(ms, gbps):
    if ms <= 0:
        return "-"
    return "%.4f<br>%.0f" % (ms, gbps)


def grid_table(rows, dtypes, dims, ceiling):
    """rows x dim table of ``ms<br>GB/s`` for each dtype."""
    out = []
    for dt in dtypes:
        by = {}
        for r in rows:
            if r.get("dtype") != dt:
                continue
            key = (int(num(r, "rows")), int(num(r, "dim")))
            if key not in by or num(r, "gbps") > num(by[key], "gbps"):
                by[key] = r
        if not by:
            continue
        present = sorted({d for (_, d) in by})
        rset = sorted({x for (x, _) in by})
        out.append("**%s** (每条目: 毫秒<br>GB/s；括号内为占实测屋顶 %s 的百分比)\n"
                   % (dt.upper(), ceiling))
        head = ["rows \\ dim"] + [str(d) for d in present]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for x in rset:
            line = [str(x)]
            for d in present:
                r = by.get((x, d))
                if r is None:
                    line.append("-")
                else:
                    pct = 100.0 * num(r, "gbps") / ceiling if ceiling else 0.0
                    line.append("%s (%.0f%%)" % (cell(num(r, "ms"), num(r, "gbps")), pct))
            out.append("| " + " | ".join(line) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musa")
    ap.add_argument("--out", default="docs/结果表格_MUSA.md")
    args = ap.parse_args()
    d = args.data

    # Section numbers are assigned in the order the sections are actually emitted,
    # so an empty data/musa/loop.csv (the serving-loop section is skipped) does not
    # leave a dangling "7b" behind it.
    sec = [-1]  # the device section is 0, so the first emitted section must come out as 0

    def h2(title):
        sec[0] += 1
        return "## %d. %s" % (sec[0], title)

    info = read_text(os.path.join(d, "info.txt"))
    roof = read_text(os.path.join(d, "roof.txt")) or info

    # The roof the tables are normalised against: the best *measured* movement of
    # the same kind this operator performs (1 read + 1 write per element).
    # The SIMT copy is the same kind of movement the operator performs (one read
    # and one write per element through the same execution pipeline), so it is the
    # denominator; the driver's memcpy path is only a fallback when the SIMT roof
    # was not measured.
    ceilings = [float(x) for x in re.findall(r"SIMT copy roof\s*:\s*([\d.]+)", roof)]
    if not ceilings:
        ceilings = [float(x) for x in re.findall(r"D2D memcpy ceiling\s*:\s*([\d.]+)", roof)]
    ceiling = max(ceilings) if ceilings else 0.0

    doc = ["# Hadamard 变换加速 —— 摩尔线程（MUSA）实验结果",
           "",
           "> 由 `tools/analyze_musa.py` 从 `data/musa/*` 自动生成，请勿手工编辑。",
           "> 屋顶口径：**同一次运行内实测的纯拷贝带宽**（`--mode roof`），"
           "而不是厂商驱动报的理论值 —— 在这块卡上后者低于实测值，详见总结报告。",
           ""]

    if info:
        doc += [h2("设备与实测屋顶"), "",
                "设备与驱动读数（`--mode info`，该模式结束时会顺带量一遍屋顶）：", "",
                "```", info.strip(), "```"]
        if roof and roof != info:
            doc += ["", "实测屋顶（`--mode roof`，下面所有百分比的基准）：", "",
                    "```", roof.strip(), "```"]
        if ceiling:
            doc += ["", "本文件所有百分比的分母 = **%.1f GB/s**（实测屋顶）。" % ceiling, ""]

    # 1. launch geometry
    tune = read_csv(os.path.join(d, "tune.csv"))
    if tune:
        doc += [h2("启动几何调优（`--mode tune`，rows=65536）"), ""]
        best = {}
        for r in tune:
            k = (r["dim"], r["dtype"])
            if k not in best or num(r, "gbps") > num(best[k], "gbps"):
                best[k] = r
        doc += ["| dim | dtype | tpr | rpb | block | grid | ms | GB/s | 占实测屋顶 |", "|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(best, key=lambda x: (int(x[0]), x[1])):
            r = best[k]
            doc.append("| %s | %s | %s | %s | %s | %s | %.4f | %.1f | %.0f%% |" % (
                r["dim"], r["dtype"], r["tpr"], r["rpb"], r["block"], r["grid"],
                num(r, "ms"), num(r, "gbps"), 100.0 * num(r, "gbps") / ceiling if ceiling else 0))
        doc += ["", "几何扫描的全部组合见 `%s`；这张表只留每个 (dim, dtype) 的最优解。" % os.path.join(d, "tune.csv"), ""]

    # 2/3. kernel matrices
    for title, fname, note in [
        ("寄存器 + warp shuffle 内核矩阵", "kernels_reg.csv", "dim ≤ 1024 的主线"),
        ("两级共享内存内核矩阵", "kernels_smem.csv", "大 head_dim，含 dim 到 16384"),
    ]:
        rows = read_csv(os.path.join(d, fname))
        if not rows:
            continue
        doc += [h2(title), "", "%s。%s" % (note, ""), ""]
        doc.append(grid_table(rows, DTYPES, DIMS, ceiling))

    # 4. in-place
    ip = read_csv(os.path.join(d, "inplace.csv"))
    if ip:
        doc += [h2("原地变换 vs 非原地（L2 容量效应）"), "",
                "| rows | dim | dtype | 工作集 | 非原地 GB/s | 原地 GB/s | 原地/非原地 |", "|---|---|---|---|---|---|---|"]
        for r in ip:
            doc.append("| %s | %s | %s | %s KiB | %.1f | %.1f | %.3f |" % (
                r["rows"], r["dim"], r["dtype"], r["footprint_kib"],
                num(r, "oop_gbps"), num(r, "ip_gbps"), num(r, "ip_over_oop")))
        doc.append("")

    # 5. fused quantisation
    qrows = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = re.match(r"quant_(fp8|int4)_(\d+)\.log$", fn)
        if not m:
            continue
        txt = read_text(os.path.join(d, fn))
        fused = re.search(r"fused\s*:\s*([\d.]+) ms", txt)
        unfused = re.search(r"unfused\s*:\s*([\d.]+) ms", txt)
        speed = re.search(r"speedup\s*:\s*([\d.]+)x", txt)
        if fused and unfused and speed:
            qrows.append((m.group(1), int(m.group(2)), float(fused.group(1)),
                          float(unfused.group(1)), float(speed.group(1))))
    if qrows:
        doc += [h2("融合量化（Hadamard + FP8/INT4）"), "",
                "| 量化 | dim | 融合 ms | 非融合 ms | 加速比 |", "|---|---|---|---|---|"]
        for q, dim, f, u, s in sorted(qrows, key=lambda x: (x[0], x[1])):
            doc.append("| %s | %d | %.4f | %.4f | %.2fx |" % (q, dim, f, u, s))
        doc.append("")

    # 6. graph vs raw
    g = read_csv(os.path.join(d, "graph.csv"))
    if g:
        doc += [h2("CUDA Graph 重放 vs 裸启动"), "",
                "| rows | dim | dtype | iters | 裸启动 ms/次 | 图重放 ms/次 | 加速比 | 裸 GB/s | 图 GB/s |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in g:
            doc.append("| %s | %s | %s | %s | %.4f | %.4f | %.3f | %.1f | %.1f |" % (
                r["rows"], r["dim"], r["dtype"], r["iters"], num(r, "raw_ms"), num(r, "graph_ms"),
                num(r, "speedup"), num(r, "raw_gbps"), num(r, "graph_gbps")))
        doc.append("")

    # 7. serving-style loop and fork width
    lp = read_csv(os.path.join(d, "loop.csv"))
    if lp:
        doc += [h2("服务场景：一趟 forward 的四种提交方式"), "",
                "A/B 量的是主机提交速率，C/D 是图重放。列含义见总结报告 7.11。", "",
                "| rows | dims | layers | plain µs | per-step µs | sequence µs | forked µs | B/D 加速比 |",
                "|---|---|---|---|---|---|---|---|"]
        for r in lp:
            doc.append("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.3f |" % (
                r["rows"], r["dims"], r["layers"], num(r, "plain_us"), num(r, "per_step_us"),
                num(r, "sequence_us"), num(r, "parallel_us"), num(r, "speedup_d")))
        doc.append("")
    for title, fname, cols in [
        ("图的根节点个数（`bench_roots.py`）", "roots.csv",
         ["layers", "nodes", "forks", "node_kind", "plain_ms", "sequence_ms", "parallel_ms", "speedup_d"]),
        ("层内分叉宽度与多流对照（`bench_group.py`）", "group.csv",
         ["layers", "tensors", "nodes", "forks", "plain_us", "sequence_us", "all_roots_us",
          "groups_us", "streams", "multi_us"]),
    ]:
        rows = read_csv(os.path.join(d, fname))
        if not rows:
            continue
        doc += [h2(title), "", "| " + " | ".join(cols) + " |",
                "|" + "---|" * len(cols)]
        for r in rows:
            doc.append("| " + " | ".join(r.get(c, "-") for c in cols) + " |")
        doc.append("")

    # 8. passes per graph
    passes = read_text(os.path.join(d, "passes.log"))
    if passes.strip():
        doc += [h2("一次提交覆盖多趟（`--passes-per-graph K`）"), "", "```",
                passes.strip()[-4000:], "```", ""]

    # 9. GEMM fusion
    gm = read_csv(os.path.join(d, "gemm.csv"))
    if gm:
        doc += [h2("旋转折进 GEMM 的 A-tile 载入"), "",
                "| M | N | K | 非融合 ms | 融合 ms | 加速比 | GFLOP/s（融合） | 省下的流量 | 逐位一致 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in gm:
            doc.append("| %s | %s | %s | %.4f | %.4f | %.4f | %.1f | %s MiB (%.0f%%) | %s |" % (
                r["m"], r["n"], r["k"], num(r, "unfused_ms"), num(r, "fused_ms"),
                num(r, "speedup"), num(r, "gflops_fused"), r["saved_mib"], num(r, "saved_pct"),
                "是" if r.get("bitwise") == "1" else "否"))
        doc.append("")

    # 10. regression suite
    tests = read_text(os.path.join(d, "tests.log"))
    if tests.strip():
        doc += [h2("回归测试"), "", "```", tests.strip(), "```", ""]

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(doc) + "\n")
    print("wrote %s (%d lines)" % (args.out, len(doc)))


if __name__ == "__main__":
    main()
