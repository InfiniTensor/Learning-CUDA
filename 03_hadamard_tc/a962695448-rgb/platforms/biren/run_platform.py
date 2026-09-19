#!/usr/bin/env python3
"""在真实 SUPA/壁仞设备构建、核查并测量；不安装或替换驱动与框架。"""
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
SOURCE_COMMIT = "1681a85ec7b832e56e672f0589a472cc8f91af95"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, env=None):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False, env=env)
    return {"command": command, "returncode": result.returncode, "output": result.stdout}


def run(command, log, stages, expected=0, env=None):
    print("RUN", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False, env=env)
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
        warp_peer = key[:-1] + (key[-1].replace("baseline_", "warp32_"),)
        if warp_peer in medians:
            comparisons.append({"shape_dtype": key[:-1], "operation": key[-1][9:], "candidate": "warp32",
                                "baseline_median_us": value, "optimized_median_us": medians[peer],
                                "warp32_median_us": medians[warp_peer], "baseline_over_warp32": value / medians[warp_peer],
                                "optimized_over_warp32": medians[peer] / medians[warp_peer]})
    return {"metric": "native SUPA events; warmup excluded; no allocation/copy in interval",
            "working_set": "same seeded read-only input reused; warm-cache timing",
            "logical_GBs_note": "logical tensor I/O estimate, not measured physical memory bandwidth",
            "statistics": summary, "comparisons_including_slowdowns": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=Path("/usr/local/birensupa/sdk/latest"))
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--output", type=Path, help="必须是新的结果目录，防止覆盖既有证据")
    parser.add_argument("--quick", action="store_true", help="仅运行快速调试矩阵，不代表完整验收")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--warp32", action="store_true", help="显式启用真实 warp32 壁仞设备的 SUPA 路径")
    parser.add_argument("--balanced-pack", action="store_true", help="Warp32 INT4 打包消融候选；须同时使用 --warp32，变换路径不变")
    parser.add_argument("--small-batch-warp", action="store_true", help="小批量 Warp32 发射消融候选；须同时使用 --warp32，仅影响 rows<=64")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if args.balanced_pack and not args.warp32:
        parser.error("--balanced-pack requires --warp32")
    if args.small_batch_warp and not args.warp32:
        parser.error("--small-batch-warp requires --warp32")
    if args.repeats < 1 or args.repeats > 10000 or args.groups < 1 or args.groups > 10000:
        parser.error("repeats/groups must be between 1 and 10000")
    sdk = args.sdk_root.resolve()
    supa = sdk / "supa"
    brcc = sdk / "brcc"
    compiler = (args.compiler or brcc / "bin/brcc").absolute()
    if not compiler.is_file():
        parser.error("SUPA brcc compiler is missing: " + str(compiler))
    # 仅配置本次子进程环境；不改系统环境、驱动或用户现有开发环境。
    runtime_env = dict(os.environ, SUPA_PATH=str(supa), BIREN_HOME=str(sdk))
    runtime_env["PATH"] = os.pathsep.join([str(brcc / "bin"), str(supa / "bin"), os.environ.get("PATH", "")])
    library_paths = [str(p) for p in (supa / "lib", brcc / "lib") if p.is_dir()]
    inherited_libraries = os.environ.get("LD_LIBRARY_PATH", "")
    runtime_env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths + ([inherited_libraries] if inherited_libraries else []))
    destination = (args.output or ROOT / "results/biren" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())).resolve()
    if destination.exists():
        parser.error("output directory already exists; select a fresh path")
    destination.mkdir(parents=True)
    binary = destination / "validate_and_benchmark"
    sources = [PLATFORM / "hadamard_api.h", PLATFORM / "hadamard_api.su",
               PLATFORM / "validate_and_benchmark.su", PLATFORM / "run_platform.py", ROOT / "include/reference.hpp"]
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": "biren", "adapted_from": {"platform": "metax", "commit": SOURCE_COMMIT},
              "quick": args.quick, "warp32_enabled": args.warp32,
              "balanced_pack_enabled": args.balanced_pack,
              "balanced_pack_scope": ["warp32_split", "warp32_fused"] if args.balanced_pack else [],
              "small_batch_warp_enabled": args.small_batch_warp,
              "small_batch_warp_scope": {"methods": ["warp32_transform", "warp32_split", "warp32_fused"],
                                          "max_rows": 64, "rows_per_block": 1, "threads_per_block": 32}
                                         if args.small_batch_warp else {},
              "sdk_environment": {key: runtime_env[key] for key in ("SUPA_PATH", "BIREN_HOME")},
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
              "git_head": capture(["git", "rev-parse", "HEAD"]),
              "git_status": capture(["git", "status", "--short"]),
              "compiler": capture([str(compiler), "--version"], env=runtime_env),
              "python_version": sys.version, "stages": []}
    manifest = destination / "run_summary.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        command = [compiler, "-x", "supa", "-std=c++17", "-O2", "-I" + str(ROOT / "include"),
                   "-I" + str(supa / "include")]
        if args.warp32:
            command.append("-DHADAMARD_BIREN_WARP32")
        if args.balanced_pack:
            command.append("-DHADAMARD_BIREN_BALANCED_PACK")
        if args.small_batch_warp:
            command.append("-DHADAMARD_BIREN_SMALL_BATCH")
        command += [PLATFORM / "hadamard_api.su", PLATFORM / "validate_and_benchmark.su",
                    "-L" + str(supa / "lib"), "-lsupa-runtime", "-Wl,-rpath," + str(supa / "lib"), "-o", binary]
        run(command, destination / "build.log", report["stages"], env=runtime_env)
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
            run([binary] + case, destination / ("invalid_%02d.log" % i), report["stages"], expected=2, env=runtime_env)
        report["cli_rejection_cases"] = len(invalid)
        command = [binary, "--validate", "--json", destination / "validation.json"]
        if args.quick:
            command.append("--quick")
        run(command, destination / "validation.log", report["stages"], env=runtime_env)
        report["validation"] = json.loads((destination / "validation.json").read_text(encoding="utf-8"))
        if report["validation"]["status"] != "PASS" or (not args.quick and not report["validation"]["full_matrix"]):
            raise RuntimeError("validation JSON does not confirm requested matrix")
        if report["validation"]["warp32_enabled"] != args.warp32:
            raise RuntimeError("validation JSON does not match requested Warp32 build")
        if report["validation"]["balanced_pack_enabled"] != args.balanced_pack:
            raise RuntimeError("validation JSON does not match requested balanced-pack build")
        if report["validation"]["small_batch_warp_enabled"] != args.small_batch_warp:
            raise RuntimeError("validation JSON does not match requested small-batch-warp build")
        if not args.no_benchmark:
            command = [binary, "--benchmark", "--csv", destination / "benchmark.csv",
                       "--groups", str(args.groups), "--repeats", str(args.repeats)]
            if args.quick:
                command += ["--batch", "1", "--seq", "17", "--heads", "1", "--dim", "128"]
            run(command, destination / "benchmark.log", report["stages"], env=runtime_env)
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
