#!/usr/bin/env python3
"""Summarize the Tensor Core vs non-Tensor-Core benchmark CSV.

Produces:
  tc_summary.md            markdown tables ready to paste into the report
  figures/tc_performance.png  4-panel comparison figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # 计算节点上可能没有 matplotlib，只输出表格
    plt = None

TRANSFORMS = ["baseline", "optimized", "warp", "tc_fast", "tc_split"]
FUSED = ["unfused_int4", "fused_opt_int4", "fused_warp_int4", "fused_tc_fast_int4",
         "fused_tc_split_int4"]
COLORS = {
    "baseline": "#9e9e9e",
    "optimized": "#1976d2",
    "warp": "#2e7d32",
    "tc_fast": "#d84315",
    "tc_split": "#6a1b9a",
    "unfused_int4": "#9e9e9e",
    "fused_opt_int4": "#1976d2",
    "fused_warp_int4": "#2e7d32",
    "fused_tc_fast_int4": "#d84315",
    "fused_tc_split_int4": "#6a1b9a",
}
# 主性能矩阵的规模，摘要表只用这一档，避免把小规模冒烟数据混进结论。
MAIN_TOKENS = 131072


def load(path: Path) -> list[dict]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def pick(rows: list[dict], dtype: str, dim: int, impl: str) -> dict | None:
    for row in rows:
        if (
            row["dtype"] == dtype
            and int(row["head_dim"]) == dim
            and row["implementation"] == impl
            and int(row["total_tokens"]) == MAIN_TOKENS
            and row["normalize"] == "true"
        ):
            return row
    return None


def markdown(rows: list[dict], gpu: str) -> str:
    dims = sorted({int(r["head_dim"]) for r in rows if int(r["total_tokens"]) == MAIN_TOKENS})
    lines = [
        "# Tensor Core Hadamard 对照结果",
        "",
        f"GPU: {gpu}；规模：{MAIN_TOKENS} tokens、normalize=true；",
        "计时为 CUDA Event，warmup 20 次、正式 100 次。",
        "",
        "## 变换 kernel 时间（avg us）",
        "",
        "| dtype | d | baseline | optimized | warp | tc_fast | tc_split | "
        "warp/optimized | tc_split/warp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dtype in ("fp16", "bf16"):
        for dim in dims:
            cells = {impl: pick(rows, dtype, dim, impl) for impl in TRANSFORMS}
            if any(value is None for value in cells.values()):
                continue
            us = {k: float(v["avg_ms"]) * 1000.0 for k, v in cells.items()}
            lines.append(
                f"| {dtype.upper()} | {dim} | "
                + " | ".join(f"{us[impl]:.2f}" for impl in TRANSFORMS)
                + f" | {us['optimized'] / us['warp']:.2f}x"
                + f" | {us['warp'] / us['tc_split']:.2f}x |"
            )

    lines += [
        "",
        "## FP64 数值诊断（采样范围见运行参数；不是 PDF 验收判据）",
        "",
        "`max_rel_error` = |err| / max(1, token 峰值)；",
        "`max_ulp` = |err| / token 峰值处的输出 dtype ULP。",
        "两者仅为诊断，不能替代绝对阈值；`max_ulp <= 0.5` 不能证明逐元素正确舍入。",
        "",
        "| dtype | d | 实现 | max_abs_error | max_rel_error | max_ulp | 与 optimized 逐位一致率 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for dtype in ("fp16", "bf16"):
        for dim in dims:
            for impl in ("warp", "tc_fast", "tc_split"):
                row = pick(rows, dtype, dim, impl)
                if row is None:
                    continue
                lines.append(
                    f"| {dtype.upper()} | {dim} | {impl} | "
                    f"{float(row['max_abs_error']):.3e} | "
                    f"{float(row['max_rel_error']):.3e} | "
                    f"{float(row['max_ulp_error']):.3f} | "
                    f"{float(row['bit_exact_vs_optimized']) * 100:.2f}% |"
                )

    lines += [
        "",
        "## 融合 INT4 端到端时间（avg us）",
        "",
        "| dtype | d | unfused | fused_opt (9.2) | fused_warp | fused_tc_fast | "
        "fused_tc_split | 最快 / unfused |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dtype in ("fp16", "bf16"):
        for dim in dims:
            cells = {impl: pick(rows, dtype, dim, impl) for impl in FUSED}
            if cells["unfused_int4"] is None or cells["fused_warp_int4"] is None:
                continue
            us = {
                k: (float(v["avg_ms"]) * 1000.0 if v is not None else None)
                for k, v in cells.items()
            }
            available = [value for value in us.values() if value is not None]
            best = min(available)
            text = " | ".join(
                "n/a" if us[impl] is None else f"{us[impl]:.2f}" for impl in FUSED
            )
            lines.append(
                f"| {dtype.upper()} | {dim} | {text} | "
                f"{us['unfused_int4'] / best:.2f}x |"
            )

    lines += [
        "",
        "## 融合正确性（与同算法 “先变换后量化” 逐位比较）",
        "",
        "| dtype | d | fused_opt (9.2) | fused_warp | fused_tc_fast | fused_tc_split |",
        "|---|---:|---|---|---|---|",
    ]
    for dtype in ("fp16", "bf16"):
        for dim in dims:
            values = []
            for impl in ("fused_opt_int4", "fused_warp_int4", "fused_tc_fast_int4",
                         "fused_tc_split_int4"):
                row = pick(rows, dtype, dim, impl)
                values.append("n/a" if row is None else row["quant_vs_unfused"])
            lines.append(f"| {dtype.upper()} | {dim} | " + " | ".join(values) + " |")
    configurations = {(r['dtype'], r['batch_size'], r['seq_len'], r['num_heads'],
                       r['head_dim'], r['normalize']) for r in rows}
    lines += ["", f"CSV 实际包含 {len(configurations)} 个唯一输入配置（不是实现行数）。"]
    if rows and 'pdf_verified' in rows[0]:
        transforms = [r for r in rows if r['kind'] == 'transform']
        lines += ["", "## 全张量绝对误差验收", "",
                  f"变换记录 {len(transforms)} 条；外部参考且绝对误差通过 "
                  f"{sum(r['pdf_verified'] == '1' for r in transforms)} 条。",
                  "未提供外部参考的 optimized 回归不计作参考库验收。"]
    else:
        lines += ["", "历史 CSV 没有全张量外部参考绝对误差字段，不能据此宣称 TC 完成 PDF 验收。"]
    lines.append("")
    return "\n".join(lines)


def figure(rows: list[dict], output: Path) -> None:
    if plt is None:
        print("[summarize_tc] matplotlib unavailable, skipping figure")
        return
    dims = sorted({int(r["head_dim"]) for r in rows if int(r["total_tokens"]) == MAIN_TOKENS})
    if not dims:
        return
    xs = range(len(dims))
    labels = [f"d={d}" for d in dims]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    # (0,0) FP16 变换时间：对数轴，否则 baseline 会把几条优化路径压成一条线。
    axis = axes[0][0]
    for impl in TRANSFORMS:
        values = [pick(rows, "fp16", dim, impl) for dim in dims]
        if any(v is None for v in values):
            continue
        axis.plot(xs, [float(v["avg_ms"]) * 1000.0 for v in values],
                  marker="o", label=impl, color=COLORS[impl])
    axis.set_yscale("log")
    axis.set_title("FP16 transform kernel time (log scale)")
    axis.set_ylabel("avg us")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(labels)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)

    # (0,1) 关键问题：Tensor Core 相对“最好的非 TC 实现”是快还是慢。
    # 比值 < 1 表示更快。基准取每个维度上 optimized 与 warp 的最小值。
    axis = axes[0][1]
    for dtype, style in (("fp16", "-"), ("bf16", "--")):
        base = []
        for dim in dims:
            candidates = [pick(rows, dtype, dim, impl) for impl in ("optimized", "warp")]
            base.append(min(float(c["avg_ms"]) for c in candidates if c is not None))
        for impl in ("tc_fast", "tc_split"):
            values = [pick(rows, dtype, dim, impl) for dim in dims]
            if any(v is None for v in values):
                continue
            axis.plot(xs, [float(v["avg_ms"]) / b for v, b in zip(values, base)],
                      style, marker="o", color=COLORS[impl], label=f"{dtype}/{impl}")
    axis.axhline(1.0, color="#424242", linewidth=1.0)
    axis.set_title("Tensor Core time / best non-TC time  (<1 = TC wins)")
    axis.set_ylabel("ratio")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(labels)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    # (1,0) FP16 融合 INT4 端到端时间：同样用对数轴。
    axis = axes[1][0]
    width = 0.16
    for offset, impl in enumerate(FUSED):
        values = [pick(rows, "fp16", dim, impl) for dim in dims]
        positions = [i + (offset - 2.0) * width for i in xs]
        heights = [float(v["avg_ms"]) * 1000.0 if v is not None else 0.0 for v in values]
        axis.bar(positions, heights, width=width, label=impl, color=COLORS[impl])
    axis.set_yscale("log")
    axis.set_title("FP16 fused INT4 end-to-end time (log scale)")
    axis.set_ylabel("avg us")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(labels)
    axis.grid(alpha=0.3, axis="y", which="both")
    axis.legend(fontsize=8)

    # (1,1) 精度。warp 与 tc_split 数值完全相同，用更大的空心标记让被覆盖的那条可见。
    axis = axes[1][1]
    markers = {"warp": ("o", 11, "none"), "tc_fast": ("s", 6, None), "tc_split": ("x", 7, None)}
    for dtype, style in (("fp16", "-"), ("bf16", "--")):
        for impl in ("warp", "tc_fast", "tc_split"):
            values = [pick(rows, dtype, dim, impl) for dim in dims]
            if any(v is None for v in values):
                continue
            marker, size, fill = markers[impl]
            axis.plot(xs, [max(float(v["max_abs_error"]), 1e-12) for v in values],
                      style, marker=marker, markersize=size,
                      markerfacecolor=fill if fill else COLORS[impl],
                      color=COLORS[impl], label=f"{dtype}/{impl}")
    axis.axhline(1e-2, color="#c62828", linestyle=":", label="FP16 threshold 1e-2")
    axis.axhline(5e-2, color="#ef6c00", linestyle=":", label="BF16 threshold 5e-2")
    axis.set_yscale("log")
    axis.set_title("max abs error vs FP64 reference (warp == tc_split)")
    axis.set_ylabel("max abs error")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(labels)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.csv)
    if not rows:
        raise SystemExit("tc result CSV is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu = rows[0]["gpu"]
    (args.output_dir / "tc_summary.md").write_text(markdown(rows, gpu), encoding="utf-8")
    figure(rows, args.output_dir / "figures" / "tc_performance.png")
    print(f"[summarize_tc] wrote {args.output_dir}/tc_summary.md and figures/tc_performance.png")


if __name__ == "__main__":
    main()
