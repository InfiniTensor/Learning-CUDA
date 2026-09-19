#!/usr/bin/env python3
"""Turn the raw `nsys stats` text dumps in docs/profile into one summary table.

For every profiled case this reports

  * how many kernel launches were traced and which kernels they were,
  * the median kernel duration,
  * the median gap between two consecutive kernels on the GPU timeline,
  * the resulting GPU occupancy over the traced window.

The gap is what a timeline view is really for: a bandwidth-bound kernel should
show up as back-to-back kernels with no bubbles, and any stall (a synchronise, a
second kernel, an un-hidden launch latency) shows up as a hole in the timeline.

Usage:  python tools/profile_summary.py [--dir docs/profile]
"""

import argparse
import os
import re

KERN_SUM = "CUDA GPU Kernel Summary"
GPU_TRACE = "CUDA GPU Trace"


def parse_kernel_summary(text):
    out = []
    if KERN_SUM not in text:
        return out
    body = text.split(KERN_SUM, 1)[1]
    body = body.split("\nProcessing [", 1)[0]  # stop before the next report
    for line in body.splitlines():
        parts = line.split(maxsplit=8)
        if len(parts) != 9:
            continue
        try:
            float(parts[0])
            int(parts[2])
        except ValueError:
            continue
        name = parts[8]
        # short C++ signature -> template head only, the demangled names are long
        m = re.search(r"fhwt::(\w+)<([^>]*)>", name)
        short = f"fhwt::{m.group(1)}<{m.group(2)}>" if m else name[:60]
        out.append({
            "instances": int(parts[2]),
            "avg_us": float(parts[3]) / 1e3,
            "med_us": float(parts[4]) / 1e3,
            "min_us": float(parts[5]) / 1e3,
            "max_us": float(parts[6]) / 1e3,
            "name": short,
        })
    return out


def parse_trace(text):
    """(start_ns, duration_ns) for every fhwt kernel launch on the timeline."""
    rows = []
    if GPU_TRACE not in text:
        return rows
    body = text.split(GPU_TRACE, 1)[1]
    for line in body.splitlines():
        if "fhwt_" not in line:
            continue
        parts = line.split()
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except (ValueError, IndexError):
            continue
    rows.sort()
    return rows


def median(xs):
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/profile")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.dir, "README.md")

    names = sorted(f[:-4] for f in os.listdir(args.dir) if f.endswith(".txt"))
    parts = ["# Nsight Systems 剖析摘要\n",
             "> 由 `tools/profile_summary.py` 从 `docs/profile/*.txt`（`nsys stats` 的原始输出）生成。\n",
             "每台机器上 `nsys` 需要管理员权限才能采集 CPU 采样与上下文切换，因此这里只使用\n"
             "GPU 侧数据：内核时长、启动间隔与时间线占空比。\n",
             "\n**同一批形状的硬件计数器（ncu）在 `docs/profile/ncu/`**：访存/SM 占峰值\n"
             "百分比、占用率、stall 分布、寄存器用量——报告 8.4 节。本工具不处理那批文件。\n"]
    parts.append("\n小规模用例（`small_iters20` / `small_iters2000`）的逐内核空隙分析见\n"
                 "`docs/profile/launch_bound_analysis.md`（由 `tools/gap_analysis.py` 从\n"
                 "nsys 导出的 sqlite 生成）——那是「这个窗口到底在等什么」的直接证据。\n"
                 "\n**图内调度的节点级证据**是三份手写的分析文件，本工具只把其中的内核汇总\n"
                 "列在下面：`loop_forkjoin_nodes.txt`（根节点个数，报告 5.8 / 8.6）、\n"
                 "`loop_group_width.txt`（每层一个分叉组的宽度 vs 串行链，报告 7.13）、\n"
                 "`loop_batch.txt`（一次提交覆盖多趟，报告 7.14）。\n"
                 "三者都用 `--cuda-graph-trace=node` 采集——默认粒度会把整张 CUDA Graph\n"
                 "当成一条不透明记录，不看节点就无从谈调度。\n")
    for name in names:
        text = open(os.path.join(args.dir, name + ".txt"), encoding="utf-8",
                    errors="ignore").read()
        ks = parse_kernel_summary(text)
        tr = parse_trace(text)
        parts.append(f"\n## `{name}`\n")
        if ks:
            parts.append("| 内核 | 启动次数 | 中位时长 (µs) | 最小 (µs) | 最大 (µs) |")
            parts.append("|---|---|---|---|---|")
            for k in ks:
                parts.append(f"| `{k['name']}` | {k['instances']} | {k['med_us']:.1f} | "
                             f"{k['min_us']:.1f} | {k['max_us']:.1f} |")
            parts.append("")
        if len(tr) > 1:
            durs = [d for _, d in tr]
            gaps = [tr[i + 1][0] - (tr[i][0] + tr[i][1]) for i in range(len(tr) - 1)]
            span = tr[-1][0] + tr[-1][1] - tr[0][0]
            # The window also contains the host-side phases between benchmark
            # bursts (data fill, allocations, other variants), so it is not a
            # measure of GPU efficiency.  What matters is how much of a *burst*
            # is spent with the GPU idle: gap / kernel-duration.
            back_to_back = 100.0 * median(gaps) / median(durs)
            parts.append(f"- 捕获窗口 {span/1e6:.1f} ms（含各阶段之间的主机侧空闲），"
                         f"共 {len(tr)} 次内核启动")
            verdict = ("稳态下 GPU 几乎没有空隙，瓶颈在内核内部而非调度"
                       if back_to_back < 5.0 else
                       "内核时长已接近启动/调度开销的量级，此时主机侧开销不可忽略")
            parts.append(f"- 内核中位时长 {median(durs)/1e3:.1f} µs，"
                         f"相邻内核中位间隔 {median(gaps)/1e3:.2f} µs "
                         f"= 内核时长的 {back_to_back:.2f}%：{verdict}")
            parts.append("")
        else:
            parts.append("- 未解析到时间线数据。\n")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
