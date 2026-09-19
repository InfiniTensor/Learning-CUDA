#!/usr/bin/env python3
"""在真实 COREX 设备构建、核查并测量；不下载/安装或替换驱动与框架。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PLATFORM = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return {"command": command, "returncode": result.returncode, "output": result.stdout}


def run(command, log, stages, expected=0):
    print("RUN", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    stages.append({"command": list(map(str, command)), "log": log.name,
                   "returncode": completed.returncode, "expected_returncode": expected,
                   "wall_seconds": time.monotonic() - started, "sha256": sha256(log)})
    print("EXIT", completed.returncode, "LOG", log, flush=True)
    if completed.returncode != expected:
        print(log.read_text(encoding="utf-8", errors="replace")[-16000:], file=sys.stderr)
        raise RuntimeError("stage failed: " + log.name)


def summarize_benchmark(path):
    samples = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ("dtype", "batch", "seq", "heads", "dim", "method"))
            samples.setdefault(key, []).append(float(row["kernel_us"]))
    medians = {}
    summary = []
    for key, values in samples.items():
        medians[key] = statistics.median(values)
        summary.append(dict(zip(("dtype", "batch", "seq", "heads", "dim", "method"), key),
                            samples=values, median_us=statistics.median(values),
                            minimum_us=min(values), maximum_us=max(values)))
    comparisons = []
    for key, value in medians.items():
        if not key[-1].startswith("baseline_"):
            continue
        peer = key[:-1] + (key[-1].replace("baseline_", "optimized_"),)
        comparisons.append({"shape_dtype": key[:-1], "operation": key[-1][9:],
                            "baseline_median_us": value, "optimized_median_us": medians[peer],
                            "baseline_over_optimized": value / medians[peer]})
        warp_peer = key[:-1] + (key[-1].replace("baseline_", "warp64_"),)
        if warp_peer in medians:
            comparisons.append({"shape_dtype": key[:-1], "operation": key[-1][9:], "candidate": "warp64",
                                "baseline_median_us": value, "optimized_median_us": medians[peer],
                                "warp64_median_us": medians[warp_peer], "baseline_over_warp64": value / medians[warp_peer],
                                "optimized_over_warp64": medians[peer] / medians[warp_peer]})
    return {"metric": "CUDA-compatible events; warmup excluded; no allocation/copy in interval",
            "working_set": "same seeded read-only input reused; warm-cache timing",
            "logical_GBs_note": "logical tensor I/O estimate, not measured physical memory bandwidth",
            "statistics": summary, "comparisons_including_slowdowns": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corex-root", type=Path, default=Path("/usr/local/corex"))
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--output", type=Path, help="必须是新的结果目录，防止覆盖既有证据")
    parser.add_argument("--quick", action="store_true", help="仅运行快速调试矩阵，不代表完整验收")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--warp64", action="store_true", help="显式启用仅供真实 warp64 设备验证的 COREX 专用路径")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 10000 or args.groups < 1 or args.groups > 10000:
        parser.error("repeats/groups must be between 1 and 10000")
    compiler = args.compiler or args.corex_root / "bin/clang++"
    if not compiler.is_file():
        parser.error("COREX compiler is missing: " + str(compiler))
    destination = (args.output or ROOT / "results/iluvatar" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())).resolve()
    if destination.exists():
        parser.error("output directory already exists; select a fresh path")
    destination.mkdir(parents=True)
    binary = destination / "validate_and_benchmark"
    sources = [PLATFORM / "hadamard_api.h", PLATFORM / "hadamard_api.cu",
               PLATFORM / "validate_and_benchmark.cu", PLATFORM / "run_platform.py", ROOT / "include/reference.hpp"]
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "quick": args.quick, "warp64_enabled": args.warp64,
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
              "git_head": capture(["git", "rev-parse", "HEAD"]),
              "git_status": capture(["git", "status", "--short"]),
              "compiler": capture([str(compiler), "--version"]),
              "python_version": sys.version, "stages": []}
    manifest = destination / "run_summary.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        command = [compiler, "-x", "ivcore", "-std=c++17", "-O2", "-I" + str(ROOT / "include"),
                   PLATFORM / "hadamard_api.cu", PLATFORM / "validate_and_benchmark.cu",
                   "-L" + str(args.corex_root / "lib"), "-Wl,-rpath," + str(args.corex_root / "lib"),
                   "-lcudart", "-o", binary]
        if args.warp64:
            command.insert(6, "-DHADAMARD_ILUVATAR_WARP64")
        run(command, destination / "build.log", report["stages"])
        report["binary_sha256"] = sha256(binary)
        # CLI 验证在创建设备上下文之前执行。错误输入必须明确返回 2。
        invalid = [[], ["--validate", "--dim", "0"], ["--validate", "--dim", "3"],
                   ["--validate", "--dim", "512"], ["--validate", "--dtype", "fp32"],
                   ["--validate", "--batch", "-1"], ["--validate", "--seq", "0"],
                   ["--validate", "--heads", "x"], ["--validate", "--batch", "18446744073709551615", "--seq", "2"],
                   ["--validate", "--heads", "999999999999999999999999999999"],
                   ["--validate", "--dim"], ["--validate", "--unknown", "1"],
                   ["--benchmark", "--repeats", "0"], ["--benchmark", "--groups", "10001"]]
        for i, case in enumerate(invalid):
            run([binary] + case, destination / ("invalid_%02d.log" % i), report["stages"], expected=2)
        report["cli_rejection_cases"] = len(invalid)
        command = [binary, "--validate", "--json", destination / "validation.json"]
        if args.quick:
            command.append("--quick")
        run(command, destination / "validation.log", report["stages"])
        report["validation"] = json.loads((destination / "validation.json").read_text(encoding="utf-8"))
        if report["validation"]["status"] != "PASS" or (not args.quick and not report["validation"]["full_matrix"]):
            raise RuntimeError("validation JSON does not confirm requested matrix")
        if report["validation"]["warp64_enabled"] != args.warp64:
            raise RuntimeError("validation JSON does not match requested Warp64 build")
        if not args.no_benchmark:
            command = [binary, "--benchmark", "--csv", destination / "benchmark.csv",
                       "--groups", str(args.groups), "--repeats", str(args.repeats)]
            if args.quick:
                command += ["--batch", "1", "--seq", "17", "--heads", "1", "--dim", "128"]
            run(command, destination / "benchmark.log", report["stages"])
            report["benchmark"] = summarize_benchmark(destination / "benchmark.csv")
        report["status"] = "PASS"
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print("FAIL", error, file=sys.stderr)
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["artifacts"] = {p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p != manifest}
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "quick": args.quick, "summary": str(manifest)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
