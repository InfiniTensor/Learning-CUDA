"""Derive baseline-to-final measurements from immutable experiment records.

The NVIDIA start and end points come from separate measurements on the same
server. Intermediate optimization ratios are never multiplied. For Hadamard,
compare the fastest measured complete pipeline available at each endpoint;
this is an explicit configuration choice, not an automatic runtime dispatcher.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = [
    ("4090d", "RTX 4090 D"),
    ("c500", "C500"),
    ("iluvatar", "MR-V100"),
    ("musa", "S4000"),
    ("ascend", "Ascend 910B2"),
]
PATHS = {
    "unfused_ms": "蝶形非融合",
    "fused_ms": "蝶形融合",
    "factorized_tc_unfused_ms": "矩阵非融合",
    "factorized_tc_fused_ms": "矩阵融合",
    "factorized_tc_materialized_ms": "矩阵写回+amax",
}


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n|"
        + "|".join(["---"] * len(headers))
        + "|\n"
        + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
    )


def fastest(metrics):
    key = min((k for k in PATHS if metrics.get(k, 0) > 0), key=metrics.get)
    return key, metrics[key]


def measurements(root, platform):
    """Join by shape, dtype and format; never equate unlike dequant outputs."""
    had = root.parent.name == "03_hadamard_tc"
    folder = "4090d/tuning" if platform == "4090d" else platform
    final = read(root / f"results/{folder}/comparison.json")["records"]
    starts = {}
    if platform == "4090d":
        for r in read(root / "results/4090d/extended.json"):
            starts[(r["rows"], r["cols"], r["dtype"], r["format"])] = (
                r["median"]["before"],
                "results/4090d/extended.json",
                "fp16" if r["format"] == "mxfp8" else "fp32",
            )
        for r in read(root / "results/4090d/before/benchmark.json"):
            if had:
                key = (r["rows"], r["dim"], r["dtype"], r["format"])
            else:
                if r["distribution"] != "normal" or r["scale_mode"] != "block":
                    continue
                key = (
                    r["median"]["rows"],
                    r["median"]["cols"],
                    r["input_dtype"],
                    r["format"],
                )
            starts[key] = (r["median"], "results/4090d/before/benchmark.json", "fp32")
    records = []
    for r in final:
        key = (r["rows"], r["cols"], r["dtype"], r["format"])
        if platform == "4090d":
            if key not in starts:
                continue  # No original baseline for this shape: do not substitute an intermediate build.
            before, source, output = starts[key]
        else:
            before, source, output = (
                r["median"]["before"],
                f"results/{folder}/comparison.json",
                "fp32",
            )
        after = r["median"]["after"]
        if had:
            bkey, b = fastest(before)
            akey, a = fastest(after)
        else:
            bkey = akey = "quant_ms"
            b, a = before[bkey], after[akey]
        records.append(
            dict(
                platform=platform,
                rows=key[0],
                cols=key[1],
                dtype=key[2],
                format=key[3],
                baseline_ms=b,
                final_ms=a,
                speedup=b / a,
                time_reduction_percent=100 * (1 - a / b),
                baseline_path=bkey,
                final_path=akey,
                baseline_source=source,
                final_source=f"results/{folder}/comparison.json",
                baseline_dequant_output=output,
                before=before,
                after=after,
            )
        )
    return records


def total_table(records, had, with_platform=False):
    same_shape = (
        with_platform
        and len({(r["rows"], r["cols"], r["dtype"]) for r in records}) == 1
    )
    same_format = with_platform and len({r["format"] for r in records}) == 1
    headers = (
        (["平台"] if with_platform else [])
        + ([] if same_shape else ["形状 / dtype"])
        + ([] if same_format else ["格式"])
    )
    if had:
        headers += ["基线方案 → 最终方案"]
    headers += ["基线 ms", "最终 ms", "总加速比", "耗时减少"]
    rows = []
    names = dict(PLATFORMS)
    for r in records:
        row = [names[r["platform"]]] if with_platform else []
        if not same_shape:
            row.append(f"{r['rows']}×{r['cols']} / {r['dtype']}")
        if not same_format:
            row.append(r["format"])
        if had:
            row += [PATHS[r["baseline_path"]] + " → " + PATHS[r["final_path"]]]
        row += [
            f"{r['baseline_ms']:.5f}",
            f"{r['final_ms']:.5f}",
            f"{r['speedup']:.2f}×",
            f"{r['time_reduction_percent']:.1f}%",
        ]
        rows.append(row)
    return table(headers, rows)


def summary(root):
    had = root.parent.name == "03_hadamard_tc"
    records = [r for platform, _ in PLATFORMS for r in measurements(root, platform)]
    result = dict(
        metric="complete_hadamard_quant_pipeline" if had else "complete_quantization",
        speedup_definition="baseline_ms / final_ms",
        reduction_definition="100 * (1 - final_ms / baseline_ms)",
        records=records,
    )
    (root / "results/performance_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    overview = [r for r in records if r["rows"] == 65536 and r["dtype"] == "fp16"]
    return records, total_table(overview, had, True)


def final_matrix_table(root, platform):
    folder = "4090d/tuning" if platform == "4090d" else platform
    rows = []
    for r in read(root / f"results/{folder}/comparison.json")["records"]:
        if r["format"] != "mxfp8" or r["rows"] == 32:
            continue
        m = r["median"]["after"]
        rows.append(
            [
                f"{r['rows']}×{r['cols']}",
                r["dtype"],
                f"{m['hadamard_ms']:.5f}",
                f"{m['factorized_tc_ms']:.5f}",
                f"{m['hadamard_ms']/m['factorized_tc_ms']:.2f}×",
            ]
        )
    return table(["形状", "dtype", "最终蝶形 ms", "最终矩阵 ms", "蝶形 / 矩阵"], rows)


def finalize_platform_report(root, platform):
    """Keep platform implementation details; standardize the reported comparisons."""
    names = {"c500": "C500", "iluvatar": "ILUVATAR", "musa": "MUSA", "ascend": "ASCEND"}
    path = root / f"REPORT_{names[platform]}.md"
    text = path.read_text()
    had = root.parent.name == "03_hadamard_tc"
    for a, b in [
        ("初始蝶形", "基线蝶形"),
        ("初始矩阵", "基线矩阵"),
        ("初始量化", "基线量化"),
        ("初始反量化", "基线反量化"),
    ]:
        text = text.replace(a, b)
    # Regeneration and repeat invocation replace, rather than append, the summary.
    text = re.sub(
        r"\n<!-- final-total-start -->.*?<!-- final-total-end -->\n",
        "\n",
        text,
        flags=re.S,
    )
    rows = [
        r
        for r in measurements(root, platform)
        if r["rows"] == 65536 and r["dtype"] == "fp16"
    ]
    intro = "\n<!-- final-total-start -->\n## 总体优化结果\n\n"
    intro += "128 MiB FP16 输入，默认块缩放、nearest。总加速比 = 基线耗时 / 最终耗时；耗时减少 = 1 − 最终耗时 / 基线耗时。\n\n"
    if had:
        intro += "比较完整“变换＋量化”，两端分别选择已测最快方案，表内明确路径。方案需显式配置；此处不表示程序自动选择。\n\n"
    else:
        intro += "比较完整量化流程，NVFP4 包含全局 amax；反量化单列，不与量化加速比混用。\n\n"
    intro += total_table(rows, had) + "\n<!-- final-total-end -->\n"
    first_section = text.index("\n## ")
    text = text[:first_section] + intro + text[first_section:]
    if had:
        text = re.sub(
            r"^\|[^\n]*(?:初始|基线|蝶形前)[^\n]*蝶形[^\n]*\n(?:\|[^\n]*\n)+",
            lambda _: final_matrix_table(root, platform),
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^\|[^\n]*蝶形前[^\n]*\n(?:\|[^\n]*\n)+",
            lambda _: final_matrix_table(root, platform),
            text,
            flags=re.M,
        )
        text = re.sub(
            r"矩阵速度与(?:初始|基线)蝶形的比值[^\n]*",
            "独立变换表比较最终实现的两种算法；总收益以完整流程表为准。",
            text,
        )
        text = re.sub(
            r"表中的矩阵 / (?:初始|基线)蝶形加速比[^\n]*",
            "独立变换表比较最终实现的两种算法；总收益以完整流程表为准。",
            text,
        )
        text = re.sub(
            r"128MiB FP16 独立变换：[^\n]*",
            "完整流程收益及最终方案见“总体优化结果”。",
            text,
        )
    text = re.sub(
        r"^!\[[^\n]*\]\(results/[^\n]+/performance.png\)\n", "", text, flags=re.M
    )
    path.write_text(re.sub(r"\n{3,}", "\n\n", text))


if __name__ == "__main__":
    summary(ROOT)
