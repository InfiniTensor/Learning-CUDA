#!/usr/bin/env python3
"""Launch-bound analysis: kernel duration vs. the gaps *between* kernels.

`tools/profile_summary.py` reads the text dumps of `nsys stats`; this tool goes
one level deeper and reads the exported sqlite (`nsys stats --force-export=true`
writes <name>.sqlite next to the <name>.nsys-rep).  That makes it possible to
correlate every kernel with the runtime API call that launched it, and to see
where the wall-clock time of a small-workload benchmark actually goes.

Motivation (report 8.5): for dim=128/rows=4096 the kernel itself is only 2.6 us,
yet a cudaEvent-bracketed window measures 15-38 us per iteration.  The timeline
explains why -- the *median* gap between two consecutive kernels is 1.5 us, but
the *mean* is ~30 us: the gap distribution is bimodal and dominated by a small
number of ~600 us submission stalls.  The measured window is ~92% host-side
waiting, so for small shapes the number says more about the submission path than
about the kernel.

Usage:
    python tools/gap_analysis.py --sqlite a.sqlite --sqlite b.sqlite \
        --out docs/profile/launch_bound_analysis.md
"""

import argparse
import os
import sqlite3
import statistics as st

US = 1e3


def load(db):
    con = sqlite3.connect(db)
    kernels = list(con.execute(
        "select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    api = {}
    for s, e, cid in con.execute(
            "select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api.setdefault(cid, []).append((s, e))
    con.close()
    return kernels, api


def analyse(label, db):
    kernels, api = load(db)
    if len(kernels) < 2:
        return None
    durs = [e - s for s, e, _ in kernels]
    gaps = [kernels[i + 1][0] - kernels[i][1] for i in range(len(kernels) - 1)]
    # Two different quantities, easy to confuse:
    #   api cost   = how long the cudaLaunchKernel *call itself* took on the CPU
    #   queue wait = kernel start - API start, i.e. how far the host was ahead
    api_cost = [e - s for _, _, cid in kernels for (s, e) in api.get(cid, [])]
    queue = [s - a for s, _, cid in kernels for (a, _) in api.get(cid, [])]
    span = kernels[-1][1] - kernels[0][0]
    return {
        "name": label,
        "kernels": len(kernels),
        "med_dur": st.median(durs),
        "sum_dur": sum(durs),
        "span": span,
        "busy": 100.0 * sum(durs) / span,
        "med_gap": st.median(gaps),
        "mean_gap": sum(gaps) / len(gaps),
        "max_gap": max(gaps),
        "med_api": st.median(api_cost) if api_cost else 0.0,
        "max_api": max(api_cost) if api_cost else 0.0,
        "med_queue": st.median(queue) if queue else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", action="append", required=True,
                    help="exported nsys sqlite file, one per profiled case")
    ap.add_argument("--out", default="docs/profile/launch_bound_analysis.md")
    args = ap.parse_args()

    rows = []
    for db in args.sqlite:
        row = analyse(os.path.splitext(os.path.basename(db))[0], db)
        if row:
            rows.append(row)

    parts = ["# 发射受限分析：内核时长 vs. 内核之间的空隙\n",
             "> 由 `tools/gap_analysis.py` 从 `nsys stats --force-export=true` 导出的\n"
             "> sqlite 生成（`CUPTI_ACTIVITY_KIND_KERNEL` / `CUPTI_ACTIVITY_KIND_RUNTIME`\n"
             "> 按 correlationId 对齐）。对应总结报告 8.5 节。\n",
             "每个用例是一个进程：`--warmup` + `--iters` 次完全相同的启动。\n"]
    if not rows:
        parts.append("(没有可解析的剖面)\n")
    else:
        parts.append("| 用例 | 内核数 | 内核中位时长 µs | 内核合计 ms | 窗口 ms | GPU 忙占比 | "
                     "间隔中位 µs | 间隔均值 µs | 间隔最大 µs | API 中位 µs | API 最大 µs | "
                     "入队等待中位 µs |")
        parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            parts.append(
                f"| `{r['name']}` | {r['kernels']} | {r['med_dur']/US:.2f} | "
                f"{r['sum_dur']/1e6:.3f} | {r['span']/1e6:.3f} | **{r['busy']:.1f}%** | "
                f"{r['med_gap']/US:.2f} | {r['mean_gap']/US:.2f} | {r['max_gap']/US:.1f} | "
                f"{r['med_api']/US:.2f} | {r['max_api']/US:.1f} | {r['med_queue']/US:.0f} |")
        parts.append("")
        parts.append("读法：`间隔中位` 贴着 1.5 µs（下一条命令已经在驱动里）而 `间隔均值` 接近 30 µs，"
                     "说明间隔分布是双峰的——均值被少量几百微秒的提交停顿支配。"
                     "`GPU 忙占比` 就是内核合计时长与窗口的比值：这个值低于 10% 时，"
                     "cudaEvent 计时窗口量到的是提交路径而不是内核。\n")
        parts.append("`API 中位/最大` 是 `cudaLaunchKernel` **调用本身**在 CPU 上的耗时：中位 6–8 µs，"
                     "远大于内核的 2.6 µs；最大十几毫秒的离群值是首次触碰新分配显存的那一次，"
                     "属于一次性噪声。\n")
        parts.append("`入队等待中位` 是 `内核开始 - API 调用开始`，也就是主机抢跑的程度。"
                     "它不是 API 开销：`iters` 越大这个值越大，说明主机把整段 burst 早早排进"
                     "队列，然后被 GPU 侧的消费速度卡住——瓶颈不在 CPU 发得不够快。\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
