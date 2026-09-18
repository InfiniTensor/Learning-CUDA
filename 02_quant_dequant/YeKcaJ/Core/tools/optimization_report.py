"""从指定原始 JSON 生成优化表；禁止挑最小值、混合操作或改用当轮复测分母。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIZES = (1048576, 4194304, 16777216)


def records(path, fmt, stage=None):
    rows = json.loads((ROOT / path).read_text())
    selected = [r for r in rows if r["format"] == fmt and r["op"] == "quant_optimized"
                and r["scope"] == "resident_gpu" and (stage is None or r.get("stage") == stage)]
    assert len(selected) == 3 and {r["elements"] for r in selected} == set(SIZES), path
    assert all(r["repeats"] == 20 for r in selected), path
    return {r["elements"]: r for r in selected}


def table(rounds, fmt, baseline_path, prefix=""):
    lines = ["| 轮次 | 元素数 | median ms | P95 ms | 相对上一轮 | 相对基线 |",
             "|---|---:|---:|---:|---:|---:|"]
    baseline = records(baseline_path, fmt)
    previous = None
    for number, (label, path, stage) in enumerate([("优化前基线", baseline_path, None)] + rounds):
        current = records(path, fmt, stage)
        for n in SIZES:
            value = current[n]["median_ms"]
            ratio = f"{previous[n]['median_ms']/value:.2f}x" if previous else "—"
            lines.append(f"| 第{number}轮：{label} | {n//1048576}M | {value:.6f} | "
                         f"{current[n]['p95_ms']:.6f} | {ratio} | {baseline[n]['median_ms']/value:.2f}x |")
        previous = current
    lines += ["", "数据来源（按轮次）：", "", f"- [优化前基线]({prefix}{baseline_path})。"]
    lines += [f"- [{label}]({prefix}{path})" + (f"，只取 stage={stage}。" if stage else "。")
              for label, path, stage in rounds]
    return "\n".join(lines)


CUDA = {
    "mxfp8": [("直接编码+融合（原实验1）", "records/03-mxfp8-fused-scale/benchmark/summary.json", None),
              ("向量化（原实验4）", "records/07-mxfp8-vectorized/summary.json", None)],
    "nvfp4": [("scale直接编码+融合（原实验2）", "records/04-nvfp4-e4m3-scale-fused/benchmark/summary.json", None),
              ("归约减少同步（原实验3）", "records/06-warp-reduction/after/benchmark/summary.json", None),
              ("partial=1024（原实验6）", "records/11-nvfp4-partials1024/repeat/summary.json", None)],
}
MUSA = [("shuffle+向量化", "records/musa-opt01/final-benchmark/summary.json", None),
        ("严格除法替代", "records/musa-opt02/recheck/summary.json", "after-divide")]
MUSA_BASE = "records/musa-opt01/baseline/summary.json"


def musa_speedup_summary():
    lines = ["## 各版本相对基线的加速比", "",
             "基线为第0轮，同格式、同元素数的 `quant_optimized / resident_gpu` median 相除。", ""]
    for fmt in ("mxfp8", "nvfp4"):
        baseline = records(MUSA_BASE, fmt)
        lines += [f"### {fmt.upper()}", "", "| 版本 | 1M | 4M | 16M |", "|---|---:|---:|---:|"]
        for number, (label, path, stage) in enumerate([("基线", MUSA_BASE, None)] + MUSA):
            current = records(path, fmt, stage)
            ratios = [f"{baseline[n]['median_ms']/current[n]['median_ms']:.2f}x" for n in SIZES]
            lines.append(f"| 第{number}轮：{label} | " + " | ".join(ratios) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    from cuda_optimization_report import render
    (ROOT / "OPTIMIZATION_LOG.md").write_text(render(), encoding="utf-8")
    musa = """# 摩尔线程优化日志

同一格式单独计算：相对上一轮 = 上一轮保存的 median / 当前 median；相对基线 = 第0轮优化前的 median / 当前 median。基线固定使用 records/musa-opt01/baseline，不使用当轮 before 复测替换。

历史条件：S4000、MUSA 5.1.0、mp_22；FP32、block/nearest；1M/4M/16M，预热3次、测量20次；量化完整 `resident_gpu` 序列，单位ms。旧记录没有实际输入哈希，仅作同配置历史趋势；新轮次遵守 [固定测试协议](BENCHMARK_PROTOCOL.md)。

"""
    for fmt in ("mxfp8", "nvfp4"):
        musa += f"## {fmt.upper()}\n\n" + table(MUSA, fmt, MUSA_BASE, "../") + "\n\n"
    musa += """## 改动与验证

- 第1轮：shuffle 归约 + float4 加载/合并写出；纯 shuffle 广播未通过测试，保留 shared scale 广播。子步骤原始测量保留在 records/musa-opt01，不另造加速链。
- 第2轮：用 FP64 商一次 RN 转回 FP32 替代高开销 __fdiv_rn，CUDA 路径未改。百万对算术测试、144组 CPU 评估、MUSA 10/10 与 CUDA 9/9 CTest通过；这些是正确性验证，不是性能数据。
- MUPTI 采集报 MT-Perf 硬件连接失败；完整设备内存/竞争检查仍缺工具，不能声明通过。

[诊断步骤](../Core/backends/musa/tools/README.md)；[前版日志快照](../records/benchmark-audit/MUSA_LOG_before_cleanup.md)。历史原始 JSON 保留，不覆盖；表格由脚本统一计算，不手填加速比。
"""
    musa += "\n" + musa_speedup_summary()
    (ROOT / "docs/MUSA_OPTIMIZATION_LOG.md").write_text(musa, encoding="utf-8")
    paths = ["records/01-before/benchmark/summary.json", "records/10-workspace-reuse/benchmark/summary.json"]
    paths += [r[1] for rounds in CUDA.values() for r in rounds] + [r[1] for r in MUSA]
    audit = []
    for path in paths:
        env_path = ROOT / Path(path).parent / "environment.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else {}
        rows = json.loads((ROOT / path).read_text())
        audit.append(dict(source=path, environment=str(env_path.relative_to(ROOT)) if env else None,
                          repeats=sorted({r.get("repeats", 0) for r in rows if r.get("scope") == "resident_gpu"}),
                          warmups=env.get("warmups"), device=env.get("gpu"),
                          actual_input_sha256=env.get("input", {}).get("sha256"),
                          source_sha256=env.get("source_sha256", {}),
                          limitation="No actual historical input SHA256; do not assert byte-identical input."))
    (ROOT / "records/benchmark-audit/conditions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
