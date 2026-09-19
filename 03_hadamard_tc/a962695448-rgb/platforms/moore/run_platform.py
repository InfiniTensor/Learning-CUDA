#!/usr/bin/env python3
"""Build and validate native MUSA Hadamard kernels on the rented Moore device."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parents[1]
BASE_COMMIT = "cb82ea6e1d5c8f78b48b6922b0f7af279696cc44"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, env):
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False, timeout=60)
    return {"command": command, "returncode": result.returncode,
            "output": result.stdout}


def run(command, log, stages, env, expected=0):
    command = list(map(str, command))
    print("RUN", log.name, flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    stages.append({"command": command, "log": log.name,
                   "returncode": result.returncode, "expected_returncode": expected,
                   "wall_seconds": time.monotonic() - started, "sha256": sha256(log)})
    print("EXIT", result.returncode, log.name, flush=True)
    if result.returncode != expected:
        print(log.read_text(errors="replace")[-12000:], file=sys.stderr)
        raise RuntimeError("Stage failed: " + log.name)


def summarize(paths):
    groups = {}
    for round_index, path in enumerate(paths, 1):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                us, ms = float(row["kernel_us"]), float(row["kernel_ms"])
                if not math.isfinite(us) or us <= 0 or not math.isclose(ms, us / 1000, rel_tol=1e-9):
                    raise ValueError("Invalid timing or millisecond conversion")
                key = (round_index, row["dtype"], row["batch"], row["seq"],
                       row["heads"], row["dim"], row["method"])
                groups.setdefault(key, []).append(us)
    medians = {key: statistics.median(values) for key, values in groups.items()}
    statistics_rows = []
    comparisons = []
    for key, samples in groups.items():
        statistics_rows.append({"round_dtype_shape_method": key, "samples_us": samples,
                                "median_us": medians[key], "median_ms": medians[key] / 1000})
        if not key[-1].startswith("baseline_"):
            continue
        operation = key[-1].removeprefix("baseline_")
        for candidate in ("optimized", "shuffle32"):
            peer = key[:-1] + (candidate + "_" + operation,)
            if peer in medians:
                comparisons.append({"round_dtype_shape": key[:-1], "operation": operation,
                                    "candidate": candidate, "baseline_us": medians[key],
                                    "candidate_us": medians[peer],
                                    "baseline_over_candidate": medians[key] / medians[peer],
                                    "time_reduction_percent": 100 * (1 - medians[peer] / medians[key])})
    return {"metric": "MUSA event interval per call, warm input; allocation/copy/validation excluded",
            "limitations": "Includes any device idle gaps caused by host submission; not application latency or cross-platform speedup",
            "rounds": len(paths), "observations": sum(map(len, groups.values())),
            "statistics": statistics_rows, "comparisons_including_slowdowns": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musa-root", type=Path, default=Path("/usr/local/musa"))
    parser.add_argument("--arch", default="mp_22")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shuffle32", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if not (1 <= args.rounds <= 10 and 1 <= args.repeats <= 10000 and 1 <= args.groups <= 10000):
        parser.error("Invalid benchmark repetition count")
    musa = args.musa_root.resolve()
    compiler = musa / "bin/mcc"
    if not compiler.is_file():
        parser.error("MUSA compiler missing: " + str(compiler))
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Select a new output directory to preserve prior results")
    destination.mkdir(parents=True)
    env = os.environ.copy()
    env["PATH"] = str(musa / "bin") + os.pathsep + env.get("PATH", "")
    env["MUSA_PATH"] = str(musa)
    env["LD_LIBRARY_PATH"] = str(musa / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    sources = [PLATFORM / name for name in ("hadamard_api.h", "hadamard_api.mu",
               "validate_and_benchmark.mu", "probe.mu", "run_platform.py", "exact_int4.hpp")]
    sources.append(ROOT / "include/reference.hpp")
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    report = {"status": "RUNNING", "platform": "moore", "target_model": "MTT S4000",
              "base_commit": BASE_COMMIT, "source_sha256": source_hashes,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "python": sys.version, "arch": args.arch, "quick": args.quick,
              "shuffle32_enabled": args.shuffle32, "scope": "probe" if args.probe_only else "validation",
              "stages": []}
    manifest = destination / "run_summary.json"

    def save():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        report["compiler"] = capture([str(compiler), "--version"], env)
        report["device_before"] = capture(["mthreads-gmi"], env)
        version = musa / "version.json"
        if version.exists():
            report["sdk_version"] = json.loads(version.read_text())
        common = [str(compiler), "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                  "--offload-arch=" + args.arch, "-I" + str(ROOT / "include")]
        link = ["-L" + str(musa / "lib"), "-lmusart"]
        probe = destination / "probe"
        run(common + [PLATFORM / "probe.mu"] + link + ["-o", probe],
            destination / "probe_build.log", report["stages"], env)
        run([probe], destination / "probe.log", report["stages"], env)
        report["probe_sha256"] = sha256(probe)
        if not args.probe_only:
            binary = destination / "validate_and_benchmark"
            defines = ["-DHADAMARD_MOORE_SHUFFLE32"] if args.shuffle32 else []
            run(common + defines + [PLATFORM / "hadamard_api.mu", PLATFORM / "validate_and_benchmark.mu"]
                + link + ["-o", binary], destination / "build.log", report["stages"], env)
            report["binary_sha256"] = sha256(binary)
            invalid = [[], ["--validate", "--dim", "0"], ["--validate", "--dim", "3"],
                       ["--validate", "--dim", "512"], ["--validate", "--dtype", "fp32"],
                       ["--validate", "--batch", "-1"], ["--validate", "--seq", "0"],
                       ["--validate", "--heads", "x"],
                       ["--validate", "--batch", "18446744073709551615", "--seq", "2"],
                       ["--validate", "--heads", "999999999999999999999999999999"],
                       ["--validate", "--dim"], ["--validate", "--unknown", "1"],
                       ["--benchmark", "--repeats", "0"], ["--benchmark", "--groups", "10001"]]
            for index, case in enumerate(invalid):
                run([binary] + case, destination / f"invalid_{index:02d}.log", report["stages"], env, expected=2)
            report["cli_rejection_cases"] = len(invalid)
            validation = [binary, "--validate", "--json", destination / "validation.json"]
            if args.quick:
                validation.append("--quick")
            run(validation, destination / "validation.log", report["stages"], env)
            result = json.loads((destination / "validation.json").read_text())
            if result["status"] != "PASS" or (not args.quick and not result["full_matrix"]):
                raise RuntimeError("Requested validation matrix was not completed")
            if result["shuffle32_enabled"] != args.shuffle32:
                raise RuntimeError("Validation build mode mismatch")
            report["validation"] = result
            save()
            if not args.no_benchmark:
                csv_paths = []
                for index in range(1, args.rounds + 1):
                    path = destination / f"benchmark_round{index}.csv"
                    command = [binary, "--benchmark", "--csv", path, "--groups", args.groups, "--repeats", args.repeats]
                    if args.quick:
                        command += ["--batch", "1", "--seq", "17", "--heads", "1", "--dim", "128"]
                    run(command, destination / f"benchmark_round{index}.log", report["stages"], env)
                    csv_paths.append(path)
                report["benchmark"] = summarize(csv_paths)
        if {str(p.relative_to(ROOT)): sha256(p) for p in sources} != source_hashes:
            raise RuntimeError("Source changed during validation")
        report["status"] = "PASS"
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print("FAIL", error, file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in destination.iterdir() if p.is_file() and p != manifest}
        save()
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "quick": args.quick, "summary": str(manifest)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
