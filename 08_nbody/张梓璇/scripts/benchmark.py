#!/usr/bin/env python3
"""按 N/kernel/chunks/Graph 运行真实 benchmark，汇总 performance.log。"""

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


KERNELS = ["baseline", "tiled", "tiled-float4", "split-source"]


def read_log(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            result[key] = value
    return result


def parse_steps_by_size(items: list[str], parser: argparse.ArgumentParser) -> dict[int, int]:
    result = {}
    for item in items:
        try:
            size_text, steps_text = item.split(":", 1)
            size, steps = int(size_text), int(steps_text)
        except ValueError:
            parser.error(f"--steps-by-size 项格式错误: {item}")
        if size <= 0 or steps <= 0:
            parser.error("--steps-by-size 中的 N 和 STEPS 必须为正")
        result[size] = steps
    return result


def steps_for_size(size: int, default_steps: int, explicit: dict[int, int], adaptive: bool) -> int:
    if size in explicit:
        return explicit[size]
    if not adaptive or size <= 4096:
        return default_steps
    # direct N-body 每步是 O(N^2)。规模翻倍时先把步数减半，
    # 同时保留至少 5 步，避免只测到初始化抖动。
    return max(5, round(default_steps * 4096 / size))


def main() -> None:
    parser = argparse.ArgumentParser(description="N-body CUDA 性能批量测试")
    parser.add_argument("executable", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--steps-by-size", nargs="*", default=[], metavar="N:STEPS",
        help="按规模覆盖步数，例如 4096:200 16384:50 65536:10",
    )
    parser.add_argument(
        "--adaptive-steps", action="store_true",
        help="N > 4096 时随规模自动减少 steps，显式 --steps-by-size 优先",
    )
    parser.add_argument("--kernel", choices=KERNELS, help="只测试一个 kernel")
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--source-chunks", type=int, nargs="+", default=[2],
        help="split-source 要扫描的 chunks，例如 1 2 4 8 16 32 64",
    )
    parser.add_argument(
        "--chunks-by-size", nargs="*", default=[], metavar="N:CHUNKS",
        help="按规模指定 split-source chunks，例如 4096:32 8192:16",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="所有测试启用 Graph")
    parser.add_argument(
        "--include-split-graph", action="store_true",
        help="split-source 同时测试 Graph off/on，其他 kernel 保持 Graph off",
    )
    parser.add_argument("--keep-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    if args.repeats <= 0 or args.steps <= 0:
        parser.error("repeats 和 steps 必须为正")
    if any(size <= 0 for size in args.sizes):
        parser.error("size 必须为正")
    if any(size <= 0 or size > 1024 for size in args.block_sizes):
        parser.error("block size 必须在 1 到 1024 之间")
    if any(chunks <= 0 or chunks > 64 for chunks in args.source_chunks):
        parser.error("source chunks 必须在 1 到 64 之间")
    if args.cuda_graph and args.include_split_graph:
        parser.error("--cuda-graph 与 --include-split-graph 不要同时使用")

    root = Path(__file__).resolve().parents[1]
    generator = root / "scripts" / "generate_particles.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    explicit_steps = parse_steps_by_size(args.steps_by_size, parser)
    chunks_by_size = parse_steps_by_size(args.chunks_by_size, parser)
    if any(chunks > 64 for chunks in chunks_by_size.values()):
        parser.error("--chunks-by-size 中的 CHUNKS 必须在 1 到 64 之间")
    kernels = [args.kernel] if args.kernel else args.kernels
    rows: list[dict[str, str]] = []

    for size in args.sizes:
        case_steps = steps_for_size(size, args.steps, explicit_steps, args.adaptive_steps)
        particles = input_dir / f"particles_{size}.txt"
        subprocess.run([sys.executable, str(generator), str(size), str(particles)], check=True)

        for kernel in kernels:
            chunk_values = args.source_chunks if kernel == "split-source" else [0]
            if kernel == "split-source" and size in chunks_by_size:
                chunk_values = [chunks_by_size[size]]
            graph_values = [args.cuda_graph]
            if kernel == "split-source" and args.include_split_graph:
                graph_values = [False, True]

            for block_size in args.block_sizes:
                for chunks in chunk_values:
                    config_chunks = chunks if chunks > 0 else args.source_chunks[0]
                    for graph_enabled in graph_values:
                        graph_tag = "graph" if graph_enabled else "plain"
                        chunk_tag = f"_c{chunks}" if kernel == "split-source" else ""
                        for repeat in range(1, args.repeats + 1):
                            case_name = (
                                f"n{size}_{kernel}_bs{block_size}{chunk_tag}_{graph_tag}_r{repeat}"
                            )
                            case_dir = args.output_dir / case_name
                            case_dir.mkdir(parents=True, exist_ok=True)
                            config = case_dir / "config.txt"
                            trajectory = case_dir / "trajectory.bin"
                            log = case_dir / "performance.log"
                            config.write_text(
                                f"dt = 1e-4\nnum_steps = {case_steps}\n"
                                f"record_interval = {case_steps}\n"
                                f"G = 1e-3\nsoftening = 1e-2\nintegrator = leapfrog\n"
                                f"kernel = {kernel}\nblock_size = {block_size}\n"
                                f"source_chunks = {config_chunks}\n"
                                f"cuda_graph = {'true' if graph_enabled else 'false'}\n",
                                encoding="utf-8",
                            )
                            command = [
                                str(args.executable), str(particles), str(config),
                                "--output", str(trajectory),
                                "--performance-log", str(log),
                                "--cpu-validation", "off", "--skip-conservation",
                                "--kernel", kernel, "--block-size", str(block_size),
                                "--source-chunks", str(config_chunks),
                            ]
                            if graph_enabled:
                                command.append("--cuda-graph")
                            print("running:", " ".join(command), flush=True)
                            subprocess.run(command, check=True)

                            row = read_log(log)
                            row["source_chunks"] = str(chunks)
                            row["cuda_graph"] = "on" if graph_enabled else "off"
                            row["repeat"] = str(repeat)
                            rows.append(row)
                            if not args.keep_trajectories:
                                trajectory.unlink(missing_ok=True)

    raw_path = args.output_dir / "benchmark.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    group_fields = (
        "particle_count", "kernel", "block_size", "source_chunks", "cuda_graph", "num_steps"
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, samples in groups.items():
        gpu_times = [float(sample["gpu_simulation_ms"]) for sample in samples]
        wall_times = [float(sample["total_simulation_wall_ms"]) for sample in samples]
        median_gpu = statistics.median(gpu_times)
        particle_count, kernel, block_size, chunks, graph_enabled, steps = key
        particle_steps = int(particle_count) * int(steps)
        throughput = particle_steps / (median_gpu / 1000.0) if median_gpu > 0.0 else 0.0
        summary_rows.append(
            {
                "N": particle_count,
                "kernel": kernel,
                "block_size": block_size,
                "source_chunks": chunks,
                "cuda_graph": graph_enabled,
                "steps": steps,
                "repeats": len(samples),
                "median_time_ms": f"{median_gpu:.6f}",
                "median_wall_time_ms": f"{statistics.median(wall_times):.6f}",
                "min_time_ms": f"{min(gpu_times):.6f}",
                "max_time_ms": f"{max(gpu_times):.6f}",
                "particle_steps_per_sec": f"{throughput:.3f}",
            }
        )

    summary_rows.sort(
        key=lambda row: (
            int(row["N"]), row["kernel"], int(row["source_chunks"]), row["cuda_graph"]
        )
    )
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"raw results: {raw_path.resolve()}")
    print(f"median summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
