#!/usr/bin/env python3
"""串行运行性能基准，保存原始 JSONL、硬件信息和可读汇总；不覆盖旧结果。"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SIZES = (1 << 20, 4 << 20, 16 << 20)
EXPECTED_INPUT_HASHES = json.loads((ROOT / "configs/benchmark-inputs-v1.json").read_text())


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixed_inputs(build, directory):
    """首次导出，随后必须匹配清单；两平台使用相同清单才能比较。"""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    expected = json.loads(manifest.read_text()) if manifest.exists() else None
    inputs = {}
    for n in SIZES:
        path = directory / f"normal_{n}.fp32"
        if not path.exists():
            if expected is not None:
                raise ValueError(f"fixed input missing: {path}; restore the frozen file")
            subprocess.run([str(build / "pipeline_mxfp8"), "--benchmark-export", str(n), str(path)], check=True)
        inputs[str(n)] = file_hash(path)
    if expected is not None and expected != inputs:
        raise ValueError("fixed benchmark input SHA256 mismatch; refusing to measure")
    if inputs != EXPECTED_INPUT_HASHES:
        raise ValueError("input differs from fixed-fp32-v1 corpus; copy the original frozen files")
    if expected is None:
        with manifest.open("x") as stream:
            json.dump(inputs, stream, indent=2)
            stream.write("\n")
    return inputs


# 调用两种格式的 pipeline --benchmark；固定 1M/4M/16M，不读取 TOML。
# 固定输入由本脚本校验；预热和计时由 benchmarks/benchmark.cu 完成。
def main():
    # ===== 1. 参数与独立结果目录 =====
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--backend", choices=["cuda", "musa"], default="cuda")
    parser.add_argument("--build-directory", type=Path, help="可选：使用自定义构建目录")
    parser.add_argument("--input-directory", type=Path, default=ROOT.parent / "input" / "benchmark-v1",
                        help="固定输入及 manifest.json；正式优化始终复用此目录")
    parser.add_argument("--exploratory", action="store_true", help="允许非 20 次试跑，不能作为正式优化轮次")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be 1..1000")
    if args.repeats != 20 and not args.exploratory:
        parser.error("正式基准固定 --repeats 20；试跑请显式添加 --exploratory")
    directory = Path(args.directory).resolve()
    # 每轮使用新目录，避免覆盖之前的 baseline 或优化记录。
    directory.mkdir(parents=True, exist_ok=False)

    # 执行只读环境查询；命令失败时抛出异常，避免生成不完整环境记录。
    def command(*cmd):
        return subprocess.check_output(cmd, text=True).strip()

    build = args.build_directory or ROOT / ("build-musa" if args.backend == "musa" else "build")
    build = build.resolve()
    inputs_dir = args.input_directory.resolve()
    input_hashes = fixed_inputs(build, inputs_dir)

    # ===== 2. 硬件、工具版本与源码标识 =====
    # 记录 Core 的所有模块及内部 CPU reference；哈希不能代替源码备份。
    metadata = dict(
        date=datetime.datetime.now().astimezone().isoformat(),
        repeats=args.repeats,
        warmups=3,
        protocol="fixed-fp32-v1",
        formal=not args.exploratory,
        input=dict(dtype="fp32", shape="1 x elements", elements=list(SIZES),
                   distribution="std::normal_distribution<float>(0,1), frozen file",
                   generator_seed=20260909, sha256=input_hashes),
        quantization=dict(scale_mode="block", rounding="nearest", seed=1234,
                          block_size=dict(mxfp8=32, nvfp4=16)),
        metric=dict(op="quant_optimized", scope="resident_gpu", statistic="median_ms",
                    p95="nearest-rank", output_types=["fp32", "fp16", "bf16"]),
        binary_sha256={fmt: file_hash(build / ("pipeline_" + fmt)) for fmt in ("mxfp8", "nvfp4")},
        backend=args.backend,
        build_directory=str(build),
        gpu=command("mthreads-gmi") if args.backend == "musa" else command(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ),
        compiler=command("/usr/local/musa/bin/mcc" if args.backend == "musa" else "nvcc", "--version"),
        cpu=command("lscpu"),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and not any(part == "build" or part.startswith("build-") for part in p.relative_to(ROOT).parts)
            and "__pycache__" not in p.parts
            and (
                p.suffix in (".cu", ".cuh", ".cpp", ".h", ".py", ".toml")
                or p.name == "CMakeLists.txt"
            )
        },
    )
    (directory / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # ===== 3. 串行执行，保存每种格式和规模的原始 JSONL =====
    # 不并行启动 GPU 任务，以免资源争用影响计时。
    records = []
    for fmt in ("mxfp8", "nvfp4"):
        binary = build / ("pipeline_" + fmt)
        for n in SIZES:
            print(f"Running {fmt}: elements={n}, repeats={args.repeats}", flush=True)
            result = subprocess.run(
                [str(binary), "--benchmark", str(n), str(args.repeats), str(inputs_dir / f"normal_{n}.fp32")],
                capture_output=True,
                text=True,
                check=True,
            )
            (directory / f"{fmt}_{n}.jsonl").write_text(result.stdout)
            records.extend(
                json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
            )
    # ===== 4. 合并 JSON 与可读报告 =====
    # resident_gpu 与 host_api 计时范围不同；后续优化应对比同名、同范围指标。
    # quant_enumeration 是内部对照，当前默认实现对应 quant_optimized。
    if fixed_inputs(build, inputs_dir) != input_hashes:
        raise ValueError("inputs changed during benchmark")
    (directory / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# 性能测试结果",
        "",
        "协议 fixed-fp32-v1：冻结 FP32 正态输入文件，seed=20260909，1M/4M/16M；block/nearest，预热3次、测量20次。输入与二进制 SHA256 见 environment.json。" if not args.exploratory else "探索性试跑：不纳入正式优化轮次。",
        f"后端：{args.backend.upper()}。不同平台的内部对照路径和编译器可能不同，性能不能直接归因于硬件。",
        f"GPU 驻留计时使用 {args.backend.upper()} event；host_api 包括量化的分配、传输和释放。",
        "host_api 中 quant_reused_workspace 复用显存和 event，不计首次创建/最终销毁；仍包含上传、量化、下载及主机结果分配/释放，与 quant_optimized 交替测量。此项衡量同进程重复调用，不代表单次 CLI 自动获得相同收益。",
        "内部对照保留 shared scale 归约与枚举元素编码；两种格式的默认 block+nearest 均使用 scale/编码融合 kernel，NVFP4 另含两级全局归约。内部对照的 scale 也可能共享优化，优化前后请比较留存的 quant_optimized 记录。",
        "中位数采用偶数样本中间两值均值，P95 使用 nearest-rank。CPU 仅测试 1M，预热一次后计时三次。",
        "",
        "| 格式 | 操作 | 范围 | 元素数 | median ms | P95 ms | 逻辑 GB/s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['format']} | {row['op']} | {row['scope']} | {row['elements']} | {row['median_ms']:.6f} | {row['p95_ms']:.6f} | {row['logical_GBps']:.3f} |"
        )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
