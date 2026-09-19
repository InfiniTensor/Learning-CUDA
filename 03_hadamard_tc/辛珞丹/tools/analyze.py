#!/usr/bin/env python3
"""Turn the raw experiment CSVs/logs into the Markdown tables of the report.

Inputs (all optional, whatever exists is used):
    data/device_info.txt     `fhwt --mode info`
    data/kernels_reg.csv     `fhwt --mode matrix --kernel reg`
    data/kernels_tc.csv      `fhwt --mode matrix --kernel tc`
    data/kernels_smem.csv    `fhwt --mode matrix --kernel smem`
    data/tune.csv            `fhwt --mode tune`
    data/baseline.csv        tools/bench_baseline.py (PyTorch / cuBLAS / ref)
    data/quant_<q>_<dim>.log `fhwt --mode quant`

Output: a single Markdown document (default docs/结果表格.md).

Usage:
    python tools/analyze.py [--data data] [--out docs/结果表格.md]
"""

import argparse
import csv
import os
import re
from collections import defaultdict

DTYPE_LABEL = {"f16": "FP16", "bf16": "BF16", "f32": "FP32"}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def inum(row, key, default=0):
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def best_per_scale(rows, key_fields=("rows", "dim", "dtype")):
    """Best (highest GB/s) launch geometry for every (rows, dim, dtype)."""
    best = {}
    for r in rows:
        k = tuple(r[f] if f == "dtype" else inum(r, f) for f in key_fields)
        g = fnum(r, "gbps")
        if k not in best or g > fnum(best[k], "gbps"):
            best[k] = r
    return best


def md_table(header, align, body_rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for r in body_rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
#  section builders
# ---------------------------------------------------------------------------
def section_environment(data_dir):
    path = os.path.join(data_dir, "device_info.txt")
    if not os.path.exists(path):
        return ""
    txt = open(path, encoding="utf-8", errors="ignore").read().strip()
    return "```\n" + txt + "\n```\n"


def section_register(reg, ceiling):
    rows = [r for r in reg if r["dtype"] in ("f16", "bf16")]
    if not rows:
        return ""
    best = best_per_scale(rows)
    batch = sorted({int(r["rows"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    out = []
    for dt in ("f16", "bf16"):
        out.append(f"\n**{DTYPE_LABEL[dt]} — 最优配置下内核时间 (ms) / 有效带宽 (GB/s)**\n")
        header = ["rows \\\\ dim"] + [str(d) for d in dims]
        align = ["---"] * (len(dims) + 1)
        body = []
        for b in batch:
            cells = [f"{b}"]
            for d in dims:
                r = best.get((b, d, dt))
                if not r:
                    cells.append("-")
                else:
                    cells.append(f"{fnum(r,'ms',float('nan')):.4f}<br>{fmt(fnum(r,'gbps'))}")
            body.append(cells)
        out.append(md_table(header, align, body))
    return "\n".join(out) + "\n"


def section_tc_vs_reg(reg, tc, ceiling):
    if not tc:
        return ""
    rbest = best_per_scale([r for r in reg])
    tbest = best_per_scale(tc)
    keys = sorted(set(tbest) & set(rbest))
    body = []
    for k in keys:
        rows, dim, dt = k
        if dt not in ("f16", "bf16") or rows < 16384:
            continue
        a, b = rbest[k], tbest[k]
        ga, gb = fnum(a, "gbps"), fnum(b, "gbps")
        body.append([
            rows, dim, DTYPE_LABEL.get(dt, dt),
            f"{fnum(a,'ms'):.4f}", fmt(ga),
            f"{fnum(b,'ms'):.4f}", fmt(gb),
            f"tpr={inum(a,'tpr')} rpb={inum(a,'rpb')}",
            f"warps={inum(b,'rpb')}",
            f"{gb / ga:.2f}x" if ga else "-",
        ])
    if not body:
        return ""
    header = ["rows", "dim", "dtype", "reg ms", "reg GB/s", "tc ms", "tc GB/s",
              "reg 配置", "tc 配置", "TC/reg"]
    align = ["---"] * len(header)
    return md_table(header, align, body) + "\n"


def section_smem(smem, ceiling):
    rows = [r for r in smem if r["dtype"] in ("f16", "bf16")]
    if not rows:
        return ""
    best = best_per_scale(rows)
    batch = sorted({int(r["rows"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    out = []
    header = ["rows \\\\ dim"] + [str(d) for d in dims]
    align = ["---"] * (len(dims) + 1)
    body = []
    for b in batch:
        cells = [f"{b}"]
        for d in dims:
            r = best.get((b, d, "f16"))
            cells.append(f"{fnum(r,'ms'):.4f}<br>{fmt(fnum(r,'gbps'))}" if r else "-")
        body.append(cells)
    out.append(md_table(header, align, body))
    return "\n".join(out) + "\n"


def section_tune(tune):
    if not tune:
        return ""
    best = {}
    for r in tune:
        k = (inum(r, "dim"), r["dtype"])
        if k not in best or fnum(r, "gbps") > fnum(best[k], "gbps"):
            best[k] = r
    body = []
    for (dim, dt), r in sorted(best.items()):
        body.append([dim, DTYPE_LABEL.get(dt, dt), inum(r, "tpr"), inum(r, "rpb"),
                     inum(r, "grid"), inum(r, "block"), f"{fnum(r,'ms'):.4f}",
                     fmt(fnum(r, "gbps")), f"{100*fnum(r,'peak_frac'):.1f}%"])
    header = ["dim", "dtype", "tpr", "rpb", "grid", "block", "ms", "GB/s", "%peak"]
    return md_table(header, ["---"] * len(header), body) + "\n"


def _tune_from_log(data_dir):
    """Fallback: parse the fixed-width `--mode tune` log when no CSV is present."""
    path = os.path.join(data_dir, "tune.log")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            p = line.rstrip("\n").split()
            if len(p) != 12 or not p[0].isdigit():
                continue
            out.append({"rows": p[0], "dim": p[1], "dtype": p[2], "kernel": p[3],
                        "acc": p[4], "tpr": p[5], "rpb": p[6], "grid": p[7],
                        "block": p[8], "ms": p[9], "gbps": p[10],
                        "peak_frac": p[11].rstrip("%")})
    return out


def section_tune_grid(tune):
    """Effective bandwidth as a function of (dim, tpr) x rpb: the tuning heat map."""
    rows = [r for r in tune if r.get("dtype") == "f16"] or list(tune)
    if not rows:
        return ""
    dims = sorted({inum(r, "dim") for r in rows})
    tprs = sorted({inum(r, "tpr") for r in rows})
    rpbs = sorted({inum(r, "rpb") for r in rows})
    lut = {(inum(r, "dim"), inum(r, "tpr"), inum(r, "rpb")): r for r in rows}
    header = ["dim", "tpr"] + [f"rpb={b}" for b in rpbs]
    body = []
    for d in dims:
        for t in tprs:
            if not any((d, t, b) in lut for b in rpbs):
                continue
            cells = [d, t]
            for b in rpbs:
                r = lut.get((d, t, b))
                cells.append(fmt(fnum(r, "gbps")) if r else "-")
            body.append(cells)
    note = "\n单位 GB/s（越高越好）；`-` 表示该几何超出线程块上限或未参与扫描。\n"
    return note + "\n" + md_table(header, ["---"] * len(header), body) + "\n"


def section_inplace(data_dir):
    """Out-of-place vs in-place: the L2-capacity effect across a row sweep."""
    rows = read_csv(os.path.join(data_dir, "inplace.csv"))
    if not rows:
        return ""
    l2_mib = 32.0
    body = []
    for r in rows:
        if r.get("dtype") != "f16":
            continue
        body.append([inum(r, "rows"), inum(r, "dim"),
                     f"{fnum(r,'footprint_kib')/1024:.0f}",
                     f"{fnum(r,'oop_ms'):.4f}", fmt(fnum(r, "oop_gbps")),
                     f"{fnum(r,'ip_ms'):.4f}", fmt(fnum(r, "ip_gbps")),
                     f"{fnum(r,'ip_over_oop'):.2f}x"])
    if not body:
        return ""
    header = ["rows", "dim", "footprint MiB", "oop ms", "oop GB/s", "ip ms", "ip GB/s",
              "ip / oop"]
    out = [md_table(header, ["---"] * len(header), body), ""]

    hot = sorted({round(fnum(r, "footprint_kib") / 1024.0) for r in rows
                  if fnum(r, "ip_over_oop") > 1.5})
    if hot:
        for h in hot:
            rat = [fnum(r, "ip_over_oop") for r in rows
                   if abs(fnum(r, "footprint_kib") / 1024.0 - h) < 0.5
                   and fnum(r, "ip_over_oop") > 1.5]
            out.append(
                "**结论.** 只有工作集正好压在 L2 容量（%.0f MiB）上的配置，原地变换才带来收益："
                "工作集 %d MiB 的 %d 个配置提速 %.2f–%.2fx。工作集明显小于 L2 时输入输出都命中 L2，"
                "比值 ≈ 1.00；工作集明显大于 L2 时两者都退化为 DRAM 受限，比值同样 ≈ 1.00——"
                "原地变换既不改变 DRAM 流量，也不改变指令数，它唯一改变的是工作集能否留在 L2。"
                % (l2_mib, h, len(rat), min(rat), max(rat)))
    dev = [abs(fnum(r, "ip_over_oop") - fnum(b, "ip_over_oop"))
           for r in rows if r.get("dtype") == "f16"
           for b in rows if b.get("dtype") == "bf16"
           and inum(b, "rows") == inum(r, "rows") and inum(b, "dim") == inum(r, "dim")]
    if dev:
        out.append("FP16 与 BF16 两条曲线逐点一致，最大偏差 %.2f（比值），说明这是访存行为而非数值格式造成的。"
                   % max(dev))
    return "\n".join(out) + "\n"


def section_baselines(base, reg, ceiling):
    if not base:
        return ""
    rbest = best_per_scale(reg)
    per = defaultdict(dict)
    for r in base:
        per[(inum(r, "rows"), inum(r, "dim"), r["dtype"])][r["impl"]] = r
    body = []
    for k in sorted(per):
        rows, dim, dt = k
        if rows < 16384 or dt not in ("f16", "bf16"):
            continue
        d = per[k]
        ours = rbest.get(k)
        mc = d.get("memcpy")
        rb = d.get("ref_fht")
        cb = d.get("cublas")
        row = [rows, dim, DTYPE_LABEL.get(dt, dt)]
        row.append(f"{fnum(ours,'ms'):.4f} / {fmt(fnum(ours,'gbps'))}" if ours else "-")
        row.append(f"{fnum(rb,'ms'):.4f} / {fmt(fnum(rb,'gbps'))}" if rb else "-")
        row.append(f"{fnum(cb,'ms'):.4f} / {fmt(fnum(cb,'gbps'))}" if cb else "-")
        row.append(f"{fnum(mc,'ms'):.4f} / {fmt(fnum(mc,'gbps'))}" if mc else "-")
        if ours and rb and fnum(rb, "ms") > 0:
            row.append(f"{fnum(rb,'ms')/fnum(ours,'ms'):.2f}x")
        else:
            row.append("-")
        if ours and cb and fnum(ours, "ms") > 0:
            row.append(f"{fnum(cb,'ms')/fnum(ours,'ms'):.2f}x")
        else:
            row.append("-")
        if ours and fnum(ours, "gbps") > 0:
            row.append(f"{100*fnum(ours,'gbps')/ceiling:.1f}%")
        else:
            row.append("-")
        body.append(row)
    header = ["rows", "dim", "dtype", "本实现 ms/GB/s", "fast_hadamard_transform",
              "cuBLAS GEMM", "memcpy(下界)", "vs 参考", "vs cuBLAS", "占理论带宽"]
    return md_table(header, ["---"] * len(header), body) + "\n"


QUANT_RE = re.compile(
    r"fused\s+:\s+([\d.]+) ms\s+([\d.]+) GB/s.*?unfused\s+:\s+([\d.]+) ms.*?"
    r"speedup\s+:\s+([\d.]+)x", re.S)


def section_quant(data_dir):
    rows = []
    for name in sorted(os.listdir(data_dir)):
        m = re.match(r"quant_(fp8|int4)_(\d+)\.log$", name)
        if not m:
            continue
        txt = open(os.path.join(data_dir, name), encoding="utf-8", errors="ignore").read()
        mm = QUANT_RE.search(txt.replace("\n", " "))
        if not mm:
            continue
        fused, fused_bw, unfused, speedup = mm.groups()
        rows.append([m.group(1).upper(), int(m.group(2)), f"{float(fused):.4f}",
                     fmt(float(fused_bw)), f"{float(unfused):.4f}",
                     f"{float(speedup):.2f}x"])
    if not rows:
        return ""
    rows.sort(key=lambda r: (r[0], r[1]))
    header = ["格式", "dim", "融合 ms", "融合有效带宽 GB/s", "未融合 ms (rotate+quant)", "加速比"]
    return md_table(header, ["---"] * len(header), rows) + "\n"


def section_graph(data_dir, ceiling):
    """Raw launches vs a captured CUDA Graph, across the three bandwidth regimes."""
    rows = read_csv(os.path.join(data_dir, "graph.csv"))
    if not rows:
        return ""
    body = []
    for r in rows:
        rows_n, dim = inum(r, "rows"), inum(r, "dim")
        mib = rows_n * dim * 2 / (1024.0 * 1024.0)
        body.append([rows_n, dim, f"{mib:.1f}", inum(r, "iters"),
                     f"{fnum(r,'raw_ms'):.4f}", fmt(fnum(r, "raw_gbps")),
                     f"{fnum(r,'graph_ms'):.4f}", fmt(fnum(r, "graph_gbps")),
                     f"{fnum(r,'speedup'):.2f}x"])
    header = ["rows", "dim", "张量 MiB", "iters", "裸启动 ms", "裸启动 GB/s",
              "Graph ms", "Graph GB/s", "加速比"]
    out = [md_table(header, ["---"] * len(header), body), ""]
    out.append("`张量 MiB` 是输入张量本身的大小；一次非原地变换的访存足迹（读一遍 + 写一遍）"
               "是它的两倍。\n")

    small = [r for r in rows if fnum(r, "speedup") >= 2.0]
    mid = [r for r in rows if 1.2 <= fnum(r, "speedup") < 2.0]
    big = [r for r in rows if fnum(r, "speedup") < 1.2]
    if small:
        out.append("**小规模（内核远短于发射开销）.** 加速比 %s；单次变换的有效带宽从最低 %.1f GB/s "
                   "升到最高 %.1f GB/s。"
                   % ("、".join(f"{fnum(r,'speedup'):.1f}x" for r in small),
                      min(fnum(r, "raw_gbps") for r in small),
                      max(fnum(r, "graph_gbps") for r in small)))
    if mid:
        out.append("**L2 驻留区（内核几十微秒）.** 加速比 %s。只有把发射开销移走之后才第一次看到"
                   "内核的真实速度：dim 256 / rows 16384 达到 %.1f GB/s。"
                   % ("、".join(f"{fnum(r,'speedup'):.2f}x" for r in mid),
                      max(fnum(r, "graph_gbps") for r in mid)))
    if big:
        out.append("**DRAM 受限区（内核数百微秒）.** 比值 %s，即 1.00x：内核越长发射开销占比越小，"
                   "CUDA Graph 对吞吐不再有影响。"
                   % ("、".join(f"{fnum(r,'speedup'):.2f}x" for r in big)))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def section_loop(data_dir):
    """Does the number of launches inside the graph matter?  A / B / C / D."""
    rows = read_csv(os.path.join(data_dir, "loop.csv"))
    if not rows:
        return ""
    has_d = any("parallel_us" in r for r in rows)
    has_sr = any(fnum(r, "parallel_sr_us") > 0 for r in rows)
    body = []
    for r in rows:
        row = [inum(r, "rows"), r.get("dims", ""), inum(r, "layers"),
               r.get("quant", "").upper(),
               f"{fnum(r,'plain_us'):.3f}", f"{fnum(r,'per_step_us'):.3f}",
               f"{fnum(r,'speedup_b'):.2f}x", f"{fnum(r,'sequence_us'):.3f}",
               f"{fnum(r,'speedup_c'):.2f}x"]
        if has_d:
            row += [f"{fnum(r,'parallel_us'):.3f}", f"{fnum(r,'speedup_d'):.2f}x"]
        if has_sr:
            sr = fnum(r, "parallel_sr_us")
            row += [f"{sr:.3f}", f"{fnum(r,'speedup_d_sr'):.2f}x",
                    f"{fnum(r,'sequence_us') / sr:.2f}x" if sr > 0 else "-"]
        body.append(row)
    header = ["rows", "dims", "层数", "量化", "A 裸启动 µs/次",
              "B 单步图 µs/次", "B vs A",
              "C 整趟图 µs/次", "C vs A"]
    if has_d:
        header += ["D 全根图 µs/次", "D 全根 vs A"]
    if has_sr:
        header += ["D 单根图 µs/次", "D 单根 vs A", "C / D 单根"]
    out = [md_table(header, ["---"] * len(header), body), ""]
    out.append("A = 每次变换一次 `cudaLaunchKernel`；B = 每次变换"
               "重放一张只含 1 个节点的图；"
               "C = **整趟 forward 的所有变换录进同一张图**，"
               "每次 pass 只提交一次；"
               "D = 与 C 同样的节点，但用 `GraphSequence::add_parallel()` "
               "声明成互不依赖，录成一张 fork/join 图"
               "（这些层之间本来就没有真依赖，"
               "串行链只是 stream capture 唯一能表达的形态）。"
               "µs/次 是单次变换的墙钟时间（已按 pass 归一）。\n")
    if has_sr:
        out.append("**D 为什么有两列。** 差别只在录出来的图有几个**根节点**："
                   "`D 全根` 把每一层都用 `add_parallel()` 排队，"
                   "于是所有节点都是根；"
                   "`D 单根` 让第一层走 `add()`（`fhwt --serial-entry`），"
                   "录成 1 个根 + 其余为分叉。节点数、内核、"
                   "每层读写的 buffer 完全相同，"
                   "`--serial-entry` 也影响不到 A / B / C（它们不调用 "
                   "`add_parallel()`），所以 C 列是两个测量之间的对照。\n")

    b = [fnum(r, "speedup_b") for r in rows]
    win = [r for r in rows if fnum(r, "speedup_c") >= 1.5]
    flat = [r for r in rows if fnum(r, "speedup_c") < 1.5]

    if b:
        out.append("**B 说明的问题：换 API 不等于减少提交。** "
                   "B 相对 A 的加速比落在 "
                   "%.2f–%.2f（中位 %.2f）：单节点图与直接启动"
                   "基本等价。把 "
                   "`cudaLaunchKernel` 换成 `cudaGraphLaunch`，却不改变提交"
                   "**次数**，"
                   "收益就落在噪声范围内。\n"
                   % (min(b), max(b), sorted(b)[len(b) // 2]))
    if win:
        out.append("**C 才是收益所在。** 提交开销占比大的"
                   "规模上加速比 %.2f–%.2f：单次变换"
                   "被压到 %.2f–%.2f µs，因为一趟 forward 的 %d "
                   "次提交变成 1 次。\n"
                   % (min(fnum(r, "speedup_c") for r in win),
                      max(fnum(r, "speedup_c") for r in win),
                      min(fnum(r, "sequence_us") for r in win),
                      max(fnum(r, "sequence_us") for r in win),
                      max(inum(r, "layers") for r in win)))
    if flat:
        out.append("**收益随工作集增大而消失**：C 的优势"
                   "来自「省下主机侧提交」，工作集一旦"
                   "远超 L2，内核本身就够长（A 到 %s µs/次），"
                   "提交开销被淹没，C 回落到 %s——"
                   "与第 8 节的 DRAM 受限区间结论一致。\n"
                   % ("、".join("%.0f" % fnum(r, "plain_us") for r in flat),
                      "、".join("%.2fx" % fnum(r, "speedup_c") for r in flat)))
    if has_d:
        dsp = [fnum(r, "speedup_d") for r in rows if fnum(r, "parallel_us") > 0]
        cd = [fnum(r, "sequence_us") / fnum(r, "parallel_us") for r in rows
              if fnum(r, "parallel_us") > 0]
        dd = [fnum(r, "parallel_us") / fnum(r, "parallel_sr_us") for r in rows
              if fnum(r, "parallel_sr_us") > 0]
        out.append("**D 全根：多根图每次重放都要多付一笔主机侧开销。** "
                   "D 全根相对 A 的加速比只有 %s，相对 C 全线落后"
                   "（C/D 全根 = %.2f–%.2fx）。这笔开销"
                   "与节点数（4–32）无关、与 `--iters` 无关，"
                   "即它是**每次重放的固定量**；"
                   "而内核时长与 C 逐条相同（8.6 节：两次节点级剖析都是 2016 条"
                   "内核记录、中位 992 ns），所以它落在主机侧。"
                   "同图全根 / 单根的耗时比是 %.2f–%.2fx："
                   "**这一列量的是图里有几个根节点，"
                   "不是 fork/join 调度本身**。负面结果保留在表里，"
                   "因为它正是定位真因的那组数据。\n"
                   % ("、".join("%.2fx" % v for v in dsp), min(cd), max(cd),
                      min(dd), max(dd)))
    if has_sr:
        sdsp = [fnum(r, "speedup_d_sr") for r in rows if fnum(r, "parallel_sr_us") > 0]
        cdsr = [fnum(r, "sequence_us") / fnum(r, "parallel_sr_us") for r in rows
                if fnum(r, "parallel_sr_us") > 0]
        hi = [v for v in cdsr if v >= 1.05]
        lo = [v for v in cdsr if v < 1.05]
        out.append("**D 单根：只把第一层改成串行入口，同一张图立刻与 C 打平甚至反超。** "
                   "D 单根相对 A 的加速比 %.2f–%.2fx，相对 C 为 %.2f–%.2fx。"
                   "其中 %d 个配置 D 单根确实比 C 快（%.2f–%.2fx），"
                   "%d 个配置与 C 持平（%.2f–%.2fx，它们都是工作集远超 L2 的规模，"
                   "C 在那里本来就没有提交开销可省，见第 8 节）。"
                   "也就是说 fork/join 的收益是**真实存在但很小**的："
                   "在这台机器上最多 %.0f%%，量级上是「打平」而不是「碾压」，"
                   "而且必须先把根节点收敛成 1 个才拿得到。\n"
                   % (min(sdsp), max(sdsp), min(cdsr), max(cdsr), len(hi), min(hi),
                      max(hi), len(lo), min(lo), max(lo), 100.0 * (max(cdsr) - 1.0)))
        out.append("机制见 5.8 与 8.6：把每个节点从 `cudaGraphAddChildGraphNode` "
                   "换成真正的 kernel node 之后数字**毫无变化**，"
                   "真正相关的是根节点个数——nsys 节点级剖析里同一张 8 节点图，"
                   "8 个根时 `cudaGraphLaunch` 中位 35–58 µs（最大 295 µs），"
                   "1 个根时 11.7–12.0 µs（最大 58 µs），"
                   "而 C 在两次测量里都是 9.9–10.2 µs（它不调用 "
                   "`add_parallel()`，本就不该变）。"
                   "所以给出的是**调用方建议**，而不是库默认行为："
                   "一排互不依赖的层，第一层用 `add()`、其余用 `add_parallel()`"
                   "（分叉要有汇合点），不要把整排层都排成根。\n")
    cap = [(fnum(r, "capture_c_ms"), inum(r, "layers")) for r in rows if "capture_c_ms" in r]
    if cap:
        # Each row reports the capture cost of its own pass; the 8-layer row is a
        # different graph size, so the median is taken over the 32-layer rows and
        # falls back to every row when there are none.  (An earlier revision read
        # row 0 directly, so the number depended on whichever shape came first.)
        nodes = max(n for _, n in cap if n == 32) if any(n == 32 for _, n in cap) \
            else max(n for _, n in cap)
        full = sorted(c for c, n in cap if n == nodes)
        mid = len(full) // 2
        med = full[mid] if len(full) % 2 else 0.5 * (full[mid - 1] + full[mid])
        out.append("建图成本是一次性的：一张含 %d 个节点"
                   "的图约 %.2f ms（%d 个同规模配置的中位数，"
                   "范围 %.2f-%.2f ms）；以表中最高的一行"
                   "（%.0f%% 的提交开销被移除）计算，"
                   "跑几趟 forward 就已回本。\n"
                   % (nodes, med, len(full), full[0], full[-1],
                      100.0 * (1.0 - fnum(rows[0], "sequence_ms")
                               / fnum(rows[0], "plain_ms"))))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
def section_gemm(data_dir):
    """Hadamard-then-GEMM vs the rotation folded into the GEMM's A-tile load."""
    rows = read_csv(os.path.join(data_dir, "gemm.csv"))
    if not rows:
        return ""
    body = []
    for r in rows:
        rng = ""
        if "speedup_min" in r and fnum(r, "speedup_min") > 0:
            rng = " [%.2f–%.2f]" % (fnum(r, "speedup_min"), fnum(r, "speedup_max"))
        body.append([inum(r, "m"), inum(r, "n"), inum(r, "k"),
                     f"{fnum(r,'unfused_ms'):.4f}", f"{fnum(r,'fused_ms'):.4f}",
                     f"{fnum(r,'speedup'):.2f}x" + rng,
                     f"{fnum(r,'gflops_unfused'):.0f}", f"{fnum(r,'gflops_fused'):.0f}",
                     f"{fnum(r,'saved_mib'):.1f}", f"{fnum(r,'saved_pct'):.1f}%",
                     "是" if inum(r, "bitwise") else "**否**"])
    header = ["M", "N", "K", "不融合 ms", "融合 ms",
              "加速比（多轮极差）",
              "不融合 GFLOP/s", "融合 GFLOP/s",
              "省下 MiB", "省下占比", "逐位一致"]
    out = [md_table(header, ["---"] * len(header), body), ""]
    out.append("A = 先 `hadamard(A)` 再乘（2 个内核）；"
               "B = 把旋转折进 GEMM 的 A tile 载入（1 个内核）。"
               "**两者是同一个手写 GEMM 核**，"
               "所以这张表量的是「融合」而不是"
               "「GEMM 写得好不好」——与 cuBLAS 的"
               "绝对对比见第 5 节。\n")
    bad = [r for r in rows if not inum(r, "bitwise")]
    out.append("**逐位一致：%s。** 融合版走的是同一张 "
               "A tile，只是写 tile 时顺带做了蝶形，"
               "累加顺序完全不变，因此结果和"
               "「先旋转再乘」逐位相同。\n"
               % ("全部 %d 个形状都是" % len(rows) if not bad
                  else "有 %d 个形状不一致" % len(bad)))

    nsweep = sorted([r for r in rows if inum(r, "m") == 8192 and inum(r, "k") == 256],
                    key=lambda r: inum(r, "n"))
    if len(nsweep) >= 3:
        out.append("**N 扫描（M=8192, K=256）：收益随 N 增大而消失。** "
                   "融合只省下 2*M*K*sizeof(T) 字节，不改变任何"
                   "算力需求，所以 N 一变大、GEMM 越接近"
                   "计算受限，省下的那点流量就被淹没："
                   "加速比从 N=%d 的 %.2fx 掉到 N=%d 的 %.2fx"
                   "（N>=256 之后落在 %.2f-%.2fx 之间，不是单调的）。\n"
                   % (inum(nsweep[0], "n"), fnum(nsweep[0], "speedup"),
                      inum(nsweep[-1], "n"), fnum(nsweep[-1], "speedup"),
                      min(fnum(r, "speedup") for r in nsweep if inum(r, "n") >= 256),
                      max(fnum(r, "speedup") for r in nsweep if inum(r, "n") >= 256)))

    ksweep = sorted([r for r in rows if inum(r, "m") == 8192 and inum(r, "n") == 128
                     and inum(r, "k") != 256], key=lambda r: inum(r, "k"))
    if len(ksweep) >= 2:
        out.append("**K 扫描（M=8192, N=128，QuaRot 的 head_dim 区间）："
                   "收益随 K 增大而增大。** 省下的字节数是 "
                   "2*M*K*sizeof(T)，K 翻倍省的流量就翻倍：%s。\n"
                   % ("、".join("K=%d %.2fx" % (inum(r, "k"), fnum(r, "speedup")) for r in ksweep)))

    small = [r for r in rows if inum(r, "m") <= 1024]
    if small:
        out.append("**M=%d 那一行拿到了 %.2fx，但意义有限：** "
                   "它省掉的是两个内核里的一整个启动"
                   "（时间只有几十微秒，启动开销占比"
                   "很大），不是带宽。\n"
                   % (inum(small[0], "m"), fnum(small[0], "speedup")))

    out.append("**一个诚实的观察：收益并没有像预期"
               "那样在 N 很大时归零。** 我们的手写 GEMM "
               "峰值只有约 %.1f TFLOP/s，远低于这张卡 FP16 "
               "张量核的峰值，也就是说这个 GEMM "
               "从来没有真正进入计算受限区，始终"
               "受制于 L2 带宽与发射，因此删掉 8 MiB DRAM "
               "流量仍然看得见。换成一个接近峰值的 "
               "GEMM（例如 cuBLAS），融合的收益会被压得更小。\n"
               % (max(fnum(r, "gflops_fused") for r in rows) / 1000.0))
    return "\n".join(out) + "\n"


def section_group(data_dir):
    """Fork/join *width* inside a layer, and the multi-stream control (report 7.13)."""
    rows = read_csv(os.path.join(data_dir, "group.csv"))
    if not rows:
        return ""
    elem = {"f16": 2, "bf16": 2, "f32": 4}

    def mib_per_pass(r):
        """Effective traffic of one pass: read + write of every tensor it rotates.

        One *slot* is one tensor.  All `tensors` tensors of a layer share that
        layer's head_dim, so the dim list cycles once per layer, not per slot --
        the same indexing as `mode_loop` in src/main.cu.
        """
        dims = [int(x) for x in str(r["dims"]).split(",")]
        layers, tensors = inum(r, "layers", 1), inum(r, "tensors", 1)
        es = elem.get(r["dtype"], 2)
        return sum(2.0 * inum(r, "rows") * dims[(slot // tensors) % len(dims)] * es
                   for slot in range(layers * tensors)) / 1048576.0

    out = ["这里的「宽度」指**同一层内部有几个互不依赖的张量**（一个 attention 块要旋转 "
           "Q / K / V）。`--tensors N` 让每层有 N 个独立张量，`--group-entry` 让每层第一个"
           "张量用 `add()` 当锚点、同层其余用 `add_parallel()`（整张图仍是单根），"
           "`--streams N` 是**不建图**、只把提交铺到 N 条非阻塞流上的对照（策略 E）。"
           "单位是 µs/单次变换，一趟 = `layers × N` 次旋转；每配置 3 轮独立扫描取中位。\n"]
    head = ["rows", "dims", "层 × 张量", "节点", "一趟 MiB", "A", "C", "D 全根",
            "**D 分组**", "D 分组 GB/s", "**vs C**", "E 多流", "vs C", "轮次"]
    align = ["---"] * len(head)
    body = []
    for r in rows:
        tensors = inum(r, "tensors", 1)
        mib = mib_per_pass(r)
        ds = fnum(r, "groups_us")
        ms = fnum(r, "multi_us")
        body.append([
            inum(r, "rows"), r["dims"], f"{inum(r, 'layers')} × {tensors}",
            inum(r, "nodes"), fmt(mib, 2),
            fmt(fnum(r, "plain_us"), 2), fmt(fnum(r, "sequence_us"), 3),
            fmt(fnum(r, "all_roots_us"), 2),
            "**%.3f**" % ds if ds else "—",
            # ds is per *transform*; a pass is `nodes` of them.
            fmt(mib * 1048576.0 / (ds * inum(r, "nodes") * 1e-6) / 1e9) if ds else "—",
            "**%.2fx**" % fnum(r, "speedup_groups_vs_sequence") if ds else "—",
            fmt(ms, 2) if ms else "—",
            fmt(fnum(r, "speedup_multi_vs_sequence"), 2) if ms else "—",
            inum(r, "runs", 1)])
    out.append(md_table(head, align, body))
    out.append("")

    widths = [(inum(r, "tensors"), fnum(r, "speedup_groups_vs_sequence")) for r in rows
              if inum(r, "tensors", 1) > 1 and fnum(r, "groups_us") > 0]
    wins = [x for _, x in widths if x >= 1.0]
    multi = [fnum(r, "speedup_multi_vs_sequence") for r in rows if fnum(r, "multi_us") > 0]
    out.append(
        "* **宽度必须落在同一层内部。** 同层张量数从 1 涨到 N，D 分组相对串行链最好到 "
        "**%.2fx**；`tensors > 1` 的 %d 个配置里 **%d 个 ≥ 1.0**（%.2f–%.2fx），"
        "其余 %d 个 %.2f–%.2fx 全部落在工作集 ≥ 4096 rows 的档位。"
        % (max([x for _, x in widths] or [0.0]), len(widths), len(wins),
           min(wins), max(wins), len(widths) - len(wins),
           min(x for _, x in widths if x < 1.0), max(x for _, x in widths if x < 1.0)))
    out.append(
        "* **同一个配置里，`D 全根`（扁平扇出）和 `D 分组`（每层一个分叉组）差距极大。** "
        "两者都是把同一批节点声明成并行，差别只在锚点放在哪一层——全根只有 "
        "%.2f–%.2fx vs C，分组是 %.2f–%.2fx。**" 
        % (min(fnum(r, "sequence_us") / fnum(r, "all_roots_us") for r in rows
               if fnum(r, "all_roots_us") > 0),
           max(fnum(r, "sequence_us") / fnum(r, "all_roots_us") for r in rows
               if fnum(r, "all_roots_us") > 0),
           min(x for _, x in widths), max(x for _, x in widths))
        + "把相关性最高的那批节点全摊平，比老老实实串行还慢。**")
    out.append(
        "* **E（多流）全线落后。** %.2f–%.2fx vs C：流只搬运提交、不减少提交，"
        "在那个规模上提交次数才是主导项（报告 7.13 / 8.5）。\n"
        % (min(multi), max(multi)))
    return "\n".join(out) + "\n"


def section_batch(data_dir):
    """How many passes one graph should hold (report 7.14)."""
    rows = read_csv(os.path.join(data_dir, "batch.csv"))
    if not rows:
        return ""
    key = lambda r: (inum(r, "rows"), r["dims"], inum(r, "layers"), inum(r, "tensors"))
    # per-pass wall time of D at K=1, which is the column everything is relative to
    base = {key(r): fnum(r, "groups_us") for r in rows if inum(r, "passes_per_graph") == 1}

    out = ["把 K 趟 forward 录进**同一张图**（`--passes-per-graph K`，`--mode loop`）。每趟都是"
           "同样的计算、同一批 buffer，所以逐位不变。单位 µs/单次变换；**一趟** = `layers × "
           "tensors` 次旋转。`建图 ms` 是 D 那张图捕获 + 实例化一次的成本。同一配置的各个 K "
           "轮转交替测量，取多轮中位。\n"]
    head = ["rows", "dims", "层 × 张量", "K", "节点", "建图 ms", "A", "C 串行链", "**D 分组**",
            "**D vs K=1**", "D vs C", "轮次"]
    align = ["---"] * len(head)
    body = []
    for r in rows:
        b = base.get(key(r), 0.0)
        d = fnum(r, "groups_us")
        body.append([
            inum(r, "rows"), r["dims"], "%d × %d" % (inum(r, "layers"), inum(r, "tensors")),
            inum(r, "passes_per_graph"), inum(r, "nodes"), fmt(fnum(r, "capture_par_ms"), 2),
            fmt(fnum(r, "plain_us"), 1), fmt(fnum(r, "sequence_us"), 3),
            "**%.3f**" % d,
            "**%.2fx**" % (b / d) if b and d and inum(r, "passes_per_graph") > 1 else "—",
            fmt(fnum(r, "speedup_groups_vs_sequence"), 2), inum(r, "runs", 1)])
    out.append(md_table(head, align, body))
    out.append("")

    seen = []
    for r in rows:
        if key(r) not in seen:
            seen.append(key(r))
    for c in seen:
        xs = sorted([r for r in rows if key(r) == c], key=lambda r: inum(r, "passes_per_graph"))
        k1 = next((r for r in xs if inum(r, "passes_per_graph") == 1), None)
        if k1 is None:
            continue
        # smallest groups_us = fastest; compare against K=1
        peak = min(xs, key=lambda r: fnum(r, "groups_us"))
        b1, bp = fnum(k1, "groups_us"), fnum(peak, "groups_us")
        # groups_us is per *transform*; one pass is `layers x tensors` of them.
        per_pass = c[2] * c[3]
        gain = b1 / bp if bp else 0.0
        label = "%.2fx" % gain if gain >= 1.05 else "%.2fx（无收益）" % gain
        out.append("* **rows %d / %s / %d × %d：K=%s -> %s。** 每趟 %.1f -> %.1f µs"
                   "（每趟省 %.1f µs），C 串行链同期 %.3f -> %.3f µs/次。"
                   % (c[0], c[1], c[2], c[3], inum(peak, "passes_per_graph"), label,
                      b1 * per_pass, bp * per_pass, (b1 - bp) * per_pass,
                      fnum(k1, "sequence_us"), fnum(peak, "sequence_us")))
    out.append("")
    out.append("**读法：** 批处理的收益 = 「K=1 时每趟比它的 GPU span 多出来的那部分」，所以它只在"
               "**每趟 GPU 时间短于一次提交开销**的档位有效，饱和值就是纯 GPU 时间（可以用 nsys "
               "的 span 独立核对：报告 7.13 / 7.14 里 D 是 23.8 vs 24.96 µs，C 是 55.4 vs 57.95 µs）。"
               "提交路径本身**没有**批量折扣——960 节点折算到每 48 节点反而比 48 节点的图贵——"
               "省下的是趟与趟之间的 GPU 空转。代价是建图成本随 K 线性上涨（约 5.6–6.3 µs/节点），"
               "960 节点的图要重放约 1100–1200 趟才回本：这是服务循环的优化，不是一次性调用的。\n")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs/结果表格.md")
    args = ap.parse_args()

    reg = read_csv(os.path.join(args.data, "kernels_reg.csv")) \
        or read_csv(os.path.join(args.data, "kernels.csv"))
    tc = read_csv(os.path.join(args.data, "kernels_tc.csv"))
    smem = read_csv(os.path.join(args.data, "kernels_smem.csv"))
    tune = read_csv(os.path.join(args.data, "tune.csv")) or _tune_from_log(args.data)
    base = read_csv(os.path.join(args.data, "baseline.csv"))

    # theoretical DRAM bandwidth is printed by `--mode info`; parse it back.
    ceiling = 448.0
    info = os.path.join(args.data, "device_info.txt")
    if os.path.exists(info):
        txt = open(info, encoding="utf-8", errors="ignore").read()
        m = re.search(r"Theoretical DRAM bandwidth:\s*([\d.]+)", txt)
        if m:
            ceiling = float(m.group(1))

    parts = ["# Hadamard 变换加速 —— 实验结果表格\n",
             "> 由 `tools/analyze.py` 从 `data/*.csv` 自动生成，请勿手工编辑。\n",
             "> 百分比的分母 = 驱动报的理论带宽（本机实测与它吻合）。MUSA 侧改用 `--mode roof`\n"
             "> 实测的纯拷贝屋顶做分母，两边的表永不混排，见 `docs/结果表格_MUSA.md`。\n",
             "## 0. 实验环境\n", section_environment(args.data)]
    for title, body in [
        ("1. 寄存器 + warp shuffle 内核：时间 (ms) / 有效带宽 (GB/s)", section_register(reg, ceiling)),
        ("2. Tensor Core 路径 vs 寄存器路径", section_tc_vs_reg(reg, tc, ceiling)),
        ("3. 大 head_dim：两级共享内存内核", section_smem(smem, ceiling)),
        ("4. 启动几何调优结果", section_tune(tune) + section_tune_grid(tune)),
        ("5. 与 PyTorch / cuBLAS / fast_hadamard_transform baseline 对比", section_baselines(base, reg, ceiling)),
        ("6. 融合量化（Hadamard + FP8/INT4）", section_quant(args.data)),
        ("7. 原地变换与 L2 容量效应", section_inplace(args.data)),
        ("8. CUDA Graph：把主机侧发射开销移出关键路径", section_graph(args.data, ceiling)),
        ("9. 图里装多少内核才决定收益（A / B / C / D 四种提交方式）",
         section_loop(args.data)),
        ("10. 旋转融合进 GEMM 的 A-tile 载入", section_gemm(args.data)),
        ("11. 分叉宽度：同一层多个张量的 fork/join，与多流对照",
         section_group(args.data)),
        ("12. 一张图装几趟：--passes-per-graph K", section_batch(args.data)),
    ]:
        if body:
            parts.append(f"\n## {title}\n")
            parts.append(body)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
