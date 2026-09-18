"""按原来的第0～6次实验生成 CUDA 日志，显式保留基准与每轮参照。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIZES = (1048576, 4194304, 16777216)
BASE = "records/01-before/benchmark/summary.json"
DIRECT = "records/02-mxfp8-direct-encode/benchmark/summary.json"
MX_FIRST = "records/03-mxfp8-fused-scale/benchmark/summary.json"
NV_FIRST = "records/04-nvfp4-e4m3-scale-fused/benchmark/summary.json"
REDUCE = "records/06-warp-reduction/after/benchmark/summary.json"
VECTOR = "records/07-mxfp8-vectorized/summary.json"
WORKSPACE = "records/10-workspace-reuse/benchmark/summary.json"
PARTIAL = "records/11-nvfp4-partials1024/repeat/summary.json"


def rows(path, fmt, op="quant_optimized", scope="resident_gpu"):
    selected = [r for r in json.loads((ROOT / path).read_text())
                if r["format"] == fmt and r["op"] == op and r["scope"] == scope]
    if (len(selected) != 3 or {r["elements"] for r in selected} != set(SIZES)
            or any(r["repeats"] != 20 for r in selected)):
        raise ValueError(f"incomplete or non-20-repeat record: {path}, {fmt}, {op}, {scope}")
    return {r["elements"]: r for r in selected}


def source(label, path):
    return f"[{label}]({path})"


def comparison(fmt, previous_path, current_path, direct=False):
    previous, current, baseline = (rows(p, fmt) for p in (previous_path, current_path, BASE))
    middle = rows(DIRECT, fmt) if direct else None
    lines = ["| 元素数 | 上一轮 median | " + ("仅直接编码 | " if direct else "")
             + "本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |",
             "|---:|---:|" + ("---:|" if direct else "") + "---:|---:|---:|---:|"]
    for n in SIZES:
        p, c, b = previous[n]["median_ms"], current[n]["median_ms"], baseline[n]["median_ms"]
        lines.append(f"| {n//1048576}M | {p:.6f} | "
                     + (f"{middle[n]['median_ms']:.6f} | " if middle else "")
                     + f"{c:.6f} | {current[n]['p95_ms']:.6f} | {p/c:.2f}x | {b/c:.2f}x |")
    return "\n".join(lines)


def speedup_summary():
    lines = ["## 各版本相对基线的加速比", "",
             "基线为第0次，同格式、同元素数的 `quant_optimized / resident_gpu` median 相除。第5次只测 host_api，不混入下表。", ""]
    versions = {
        "mxfp8": [("第0次：基线", BASE), ("第1次中间步骤：仅直接编码", DIRECT),
                  ("第1次：直接编码 + scale 融合", MX_FIRST), ("第4次：向量化（最终）", VECTOR)],
        "nvfp4": [("第0次：基线", BASE), ("第2次：scale 编码 + 归约融合", NV_FIRST),
                  ("第3次：归约减少同步", REDUCE), ("第6次：partial=1024（最终）", PARTIAL)],
    }
    for fmt, entries in versions.items():
        baseline = rows(BASE, fmt)
        lines += [f"### {fmt.upper()}", "", "| 版本 | 1M | 4M | 16M |", "|---|---:|---:|---:|"]
        for label, path in entries:
            current = rows(path, fmt)
            ratios = [f"{baseline[n]['median_ms']/current[n]['median_ms']:.2f}x" for n in SIZES]
            lines.append(f"| {label} | " + " | ".join(ratios) + " |")
        lines.append("")
    return "\n".join(lines)


def render():
    lines = ["# CUDA 优化日志", "",
             "保留原第0～6次编号。数值从下方链接的原始 JSON 提取；不把新重测替换成历史成绩。单位均为 ms，1M=1,048,576。", "",
             "历史记录声明的条件：RTX 3060 Laptop，FP32，block/nearest，MXFP8 block=32、NVFP4 block=16，预热3次、测量20次。量化表取 `quant_optimized / resident_gpu` 的 median/P95；NVFP4 包含全局归约。第5次单列 host_api。", "",
             "加速比只有两列：相对上一轮＝上一轮已保存 median ÷ 本轮 median；相对基线＝第0次同格式、同元素数的 median ÷ 本轮 median。基线固定，不随优化轮次改变。", "",
             "旧实验未保存实际输入文件哈希，下面是历史记录对照；不能补称已逐字节控制输入。今后执行 [固定输入协议](docs/BENCHMARK_PROTOCOL.md)。", "",
             "## 第0次：优化前基准（2026-09-10）", "",
             "这是两种格式最初的参照，不能省略，也不是内部 `quant_enumeration` 的耗时。", "",
             "| 格式 | 元素数 | 量化 median | 量化 P95 | FP32反量化 | FP16反量化 | BF16反量化 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for fmt in ("mxfp8", "nvfp4"):
        quant = rows(BASE, fmt)
        dequant = [rows(BASE, fmt, "dequant_" + t) for t in ("fp32", "fp16", "bf16")]
        for n in SIZES:
            values = [quant[n]["median_ms"], quant[n]["p95_ms"]] + [d[n]["median_ms"] for d in dequant]
            lines.append(f"| {fmt.upper()} | {n//1048576}M | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "来源：" + source("第0次完整数据", BASE) + "。", "",
              "## 第1次：MXFP8 直接编码 + scale 融合（2026-09-10）", "",
              "nearest 直接生成 E4M3；再将分组归约、scale 计算和编码融合。保留当时两步分别测得的数据。", "",
              "上一轮＝第0次；基线＝第0次。", "",
              comparison("mxfp8", BASE, MX_FIRST, direct=True), "",
              "来源：" + source("上一轮", BASE) + "、" + source("仅直接编码", DIRECT) + "、" + source("本轮最终结果", MX_FIRST) + "。", "",
              "## 第2次：NVFP4 scale 直接编码 + 归约融合（2026-09-11）", "",
              "block scale 使用直接 E4M3 编码，再融合分组归约与打包编码；全局最大值归约仍保留。", "",
              "上一轮＝第0次 NVFP4；基线＝第0次 NVFP4。", "",
              comparison("nvfp4", BASE, NV_FIRST), "",
              "仅 scale 直接编码的中间步骤：旧日志只记录4M为0.238592 ms；1M/16M未单独测量。该4M数字尚未定位到独立原始 JSON，仅保留为旧日志记载，不参与正式加速比计算。", "",
              "来源：" + source("上一轮", BASE) + "、" + source("本轮最终结果", NV_FIRST) + "；中间步骤见 " + source("原日志第2次", "records/benchmark-audit/CUDA_LOG_before_cleanup.md") + "。", "",
              "## 第3次：NVFP4 全局归约减少同步（2026-09-14）", "",
              "使用 warp shuffle 归约，再合并各 warp 结果，减少整块同步与共享内存。", "",
              "上一轮＝第2次；基线＝第0次 NVFP4。", "",
              comparison("nvfp4", NV_FIRST, REDUCE), "",
              "来源：" + source("基线", BASE) + "、" + source("上一轮", NV_FIRST) + "、" + source("本轮", REDUCE) + "。当轮另测的 before 不替换上一轮保存值。", "",
              "## 第4次：MXFP8 向量化加载与打包写回（2026-09-14）", "",
              "float4 加载，四个 E4M3 合并为32位写回；8线程处理一个32元素分组，尾部回退标量。", "",
              "上一轮＝第1次 MXFP8；基线＝第0次 MXFP8。第2、3次针对NVFP4，不改写MXFP8的参照。", "",
              comparison("mxfp8", MX_FIRST, VECTOR), "",
              "来源：" + source("基线", BASE) + "、" + source("上一轮", MX_FIRST) + "、" + source("本轮", VECTOR) + "。", "",
              "## 第5次：workspace 复用对照（2026-09-14）", "",
              "本次测 host_api：同一进程每次新建/释放 workspace 与复用 workspace 对照，包含上传、量化、下载。kernel 未改。保留原实验数据，但不与 resident_gpu 基线混算加速比。", "",
              "| 格式 | 元素数 | 每次新建 median | 复用 median | 复用 P95 |",
              "|---|---:|---:|---:|---:|"]
    for fmt in ("mxfp8", "nvfp4"):
        fresh = rows(WORKSPACE, fmt, scope="host_api")
        reused = rows(WORKSPACE, fmt, op="quant_reused_workspace", scope="host_api")
        for n in SIZES:
            lines.append(f"| {fmt.upper()} | {n//1048576}M | {fresh[n]['median_ms']:.6f} | "
                         f"{reused[n]['median_ms']:.6f} | {reused[n]['p95_ms']:.6f} |")
    lines += ["", "来源：" + source("同批次新建与复用数据", WORKSPACE) + "。", "",
              "## 第6次：NVFP4 partial 数量调优（2026-09-14）", "",
              "全局归约 partial block 上限从4096降为1024，沿用 grid-stride loop；scale 公式和编码不变。", "",
              "上一轮＝第3次 NVFP4；基线＝第0次 NVFP4。采用原日志指定的 repeat 整组数据，未逐尺寸择优。", "",
              comparison("nvfp4", REDUCE, PARTIAL), "",
              "来源：" + source("基线", BASE) + "、" + source("上一轮", REDUCE) + "、" + source("本轮", PARTIAL) + "。原日志用当轮 before 算的1.18x/1.02x/1.01x，不是这里“相对上一轮保存值”的参照。", "",
              "完整诊断与正确性记录见 " + source("原日志备份", "records/benchmark-audit/CUDA_LOG_before_cleanup.md") + "。最近的跨平台重测单独保存在 records/controlled-v1/，不插入这六次实验，也不替代上述基准。", ""]
    lines += [speedup_summary()]
    return "\n".join(lines)


if __name__ == "__main__":
    (ROOT / "OPTIMIZATION_LOG.md").write_text(render(), encoding="utf-8")
