# =============================================================================
#  ncu_summary.py -- 把 docs/profile/ncu/*.txt 汇总成一张 markdown 表
#
#  输入是 scripts/collect_ncu.ps1 采下来的原始文件（ncu 的 --csv 输出，一次运行
#  一个文件，每行是「section / metric / value」）。本工具只做一件事：把下面这几个
#  读数按用例拉出来排成表，方便核对 docs/profile/ncu/README.md 里的数字与原始文件
#  是否一致——表格与原始数据不一致是这类文档最容易出的错。
#
#  用法：
#      python tools/ncu_summary.py --dir docs/profile/ncu [--out docs/profile/ncu/汇总.md]
# =============================================================================
import argparse
import csv
import io
import os

# 表里要的列：(显示名, ncu 的 Metric Name)
FIELDS = [
    ("DRAM 吞吐", "DRAM Throughput"),
    ("SM 吞吐", "Compute (SM) Throughput"),
    ("L1/TEX", "L1/TEX Cache Throughput"),
    ("L2", "L2 Cache Throughput"),
    ("占用率", "Achieved Occupancy"),
    ("寄存器/线程", "Registers Per Thread"),
    ("waves/SM", "Waves Per SM"),
    ("No Eligible", "No Eligible"),
    ("Issued/sched", "Issued Warp Per Scheduler"),
]


def read_one(path):
    """返回 (kernel, grid, block, {metric: value}, [(stall, value)...])。"""
    lines = [l for l in io.open(path, encoding="utf-8", errors="replace").read().splitlines()
             if l.startswith('"')]
    if not lines:
        return None
    rows = list(csv.reader(lines))
    hdr, body = rows[0], rows[1:]
    idx = {h: i for i, h in enumerate(hdr)}
    need = idx["Metric Value"]
    got = {}
    stalls = []
    kernel = grid = block = ""
    for r in body:
        if len(r) <= need:
            continue
        name, val = r[idx["Metric Name"]], r[idx["Metric Value"]]
        if not kernel:
            kernel, grid, block = r[idx["Kernel Name"]], r[idx["Grid Size"]], r[idx["Block Size"]]
        if name not in got:
            got[name] = val
        if name.startswith("Stall "):
            try:
                stalls.append((name, float(val.replace(",", ""))))
            except ValueError:
                pass
    return kernel, grid, block, got, stalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/profile/ncu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.dir, "汇总.md")

    names = sorted(f[:-4] for f in os.listdir(args.dir)
                   if f.endswith(".txt") and not f.startswith("_"))
    parts = ["# ncu 读数汇总（由 `tools/ncu_summary.py` 生成）\n",
             "\n与 `docs/profile/ncu/README.md` 里的表同源；数值直接取自原始文件的 `Metric Name` / `Metric Value` 两列。\n",
             "\n| 用例 | 内核 | grid × block | " + " | ".join(f[0] for f in FIELDS) + " | 主 stall | 时长 |",
             "|---|---|---|" + "---|" * (len(FIELDS) + 2)]
    for n in names:
        rec = read_one(os.path.join(args.dir, n + ".txt"))
        if not rec:
            continue
        kernel, grid, block, got, stalls = rec
        short = kernel.split("fhwt::")[-1].split("<")[0] if "fhwt" in kernel else kernel
        dur = got.get("Duration", "")
        try:
            dur = "%.1f µs" % (float(dur.replace(",", "")) / 1000.0)
        except ValueError:
            pass
        cells = ["%.2f%%" % float(got[f[1]]) if f[1] in got and f[1].endswith("Throughput") or
                 (f[1] in got and f[1] in ("Achieved Occupancy", "No Eligible"))
                 else got.get(f[1], "") for f in FIELDS]
        top = ""
        if stalls:
            stalls.sort(key=lambda x: -x[1])
            tot = sum(v for _, v in stalls) or 1.0
            top = "%s %.1f%%" % (stalls[0][0].replace("Stall ", ""), 100.0 * stalls[0][1] / tot)
        parts.append("| `%s` | %s | %s × %s | %s | %s | %s |" % (
            n, short, grid.strip("()").split(",")[0], block.strip("()").split(",")[0],
            " | ".join(cells), top, dur))
    parts.append("")
    io.open(out_path, "w", encoding="utf-8", newline="").write("\n".join(parts))
    print("wrote", out_path, "(%d cases)" % len(names))


if __name__ == "__main__":
    main()
