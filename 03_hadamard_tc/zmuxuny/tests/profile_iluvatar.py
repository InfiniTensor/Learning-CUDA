"""Collect CoreX validation/profiling evidence separately from event benchmarks.

Each invocation records its command, exit code and the exact binary hash. ixsys
traces are native IXplorer files; ixkn output includes replayed hardware metrics.
A successful sanitizer exit alone is insufficient: require a zero-error summary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from generate import generate
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/iluvatar" / ("hadamard" if HADAMARD else "quantize"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/iluvatar/profile")
    parser.add_argument("--mode", choices=["sanitizer", "trace", "counters", "all"], default="all")
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corex = Path(os.environ.get("COREX_PATH", "/usr/local/corex"))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(corex / "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
    records = []
    report = {"gpu": "Iluvatar MR-V100", "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "timing_note": "Instrumented runs; use uninstrumented device-event benchmarks for speedups.", "runs": records}
    modes = ["sanitizer", "trace", "counters"] if args.mode == "all" else [args.mode]
    with tempfile.TemporaryDirectory(prefix="iluvatar-profile-") as tmp:
        temp = Path(tmp)
        for mode in modes:
            for dtype in ([1, 2] if HADAMARD else [0, 1]):
                # Sanitizer cases cover tails, both formats and both input dtypes.
                if mode == "sanitizer":
                    rows, cols = ((19, 1024) if dtype == 1 else (31, 128)) if HADAMARD else ((3, 35) if dtype == 0 else (65, 1024))
                else:
                    rows, cols = (8192, 1024) if HADAMARD else (1024, 1024)
                write_tensor(temp / "input", generate(rows, cols, "normal"), dtype)
                for fmt in ["mxfp8", "nvfp4"]:
                    label = f"{mode}_{fmt}_{['fp32','fp16','bf16'][dtype]}"
                    dest = output / label
                    dest.mkdir(exist_ok=True)
                    command = [str(binary), "--input", str(temp / "input"), "--output", str(temp / "out"),
                               "--packed", str(temp / "packed"), "--log", str(dest / "metrics.json"),
                               "--repeats", "3" if mode == "trace" else "1"]
                    if HADAMARD:
                        command += ["--format", fmt, "--materialized_compare", "1"]
                        if mode == "sanitizer":
                            command += ["--random_sign", "1", "--rounding", "stochastic"]
                    else:
                        command += ["--config", str(ROOT / "configs" / f"{fmt}.toml")]
                    if mode == "sanitizer":
                        tools = [(name, [str(corex / "bin/ixsan"), "--tool", name]) for name in ["memcheck", "racecheck", "initcheck"]]
                    elif mode == "trace":
                        tools = [("ixsys", [str(corex / "bin/ixsys"), "-t", "cuda", "--print-gpu-summary", "-o", str(dest / "trace.ixsys")])]
                    else:
                        # ixkn splits literal commas into separate filters; use a
                        # digit-delimited regex and verify the actually captured name.
                        if HADAMARD:
                            patterns = [("matrix_transform", f"hadamard_mma_kernel<{cols}[^0-9]+{dtype}[^0-9]+0[^0-9]+"),
                                        ("matrix_fused", f"hadamard_mma_kernel<{cols}[^0-9]+{dtype}[^0-9]+2[^0-9]+")]
                            if fmt == "nvfp4":
                                patterns += [("matrix_amax", f"hadamard_mma_kernel<{cols}[^0-9]+{dtype}[^0-9]+1[^0-9]+"),
                                             ("matrix_materialized", f"hadamard_mma_kernel<{cols}[^0-9]+{dtype}[^0-9]+3[^0-9]+")]
                        else:
                            patterns = [("quant_vector_kernel", "quant_vector_kernel"),
                                        ("dequant_vector_kernel", "dequant_vector_kernel")]
                            if fmt == "nvfp4":
                                patterns.append(("maximum", "maximum"))
                        tools = [(name, [str(corex / "bin/ixkn-cli"), "--kernel-name", pattern,
                                         "--launch-count", "1", "--set", "full"])
                                 for name, pattern in patterns]
                    for name, prefix in tools:
                        proc = subprocess.run(prefix + command, capture_output=True, text=True, env=env, timeout=120)
                        log = proc.stdout + proc.stderr
                        logfile = dest / f"{name}.log"
                        logfile.write_text(log)
                        zero_summary = bool(re.search(
                            r"RACECHECK SUMMARY:\s*0 hazards displayed \(0 errors, 0 warnings\)"
                            if name == "racecheck" else r"ERROR SUMMARY:\s*0 errors", log))
                        passed = proc.returncode == 0 and (mode != "sanitizer" or zero_summary)
                        if mode == "trace":
                            passed = passed and (dest / "trace.ixsys").is_file() and "GPU activities:" in log
                        elif mode == "counters":
                            captured = re.findall(r"^\s*Kernel:\s*(.+)$", log, re.MULTILINE)
                            pattern = prefix[prefix.index("--kernel-name") + 1]
                            passed = passed and bool(captured) and all(re.search(pattern, k) for k in captured)
                        records.append({"case": label, "tool": name, "shape": [rows, cols], "dtype": dtype,
                                        "command": prefix + command, "returncode": proc.returncode,
                                        "zero_error_summary": zero_summary if mode == "sanitizer" else None,
                                        "passed": passed, "log": str(logfile.relative_to(output))})
                        (output / f"summary_{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(label, name, "PASS" if passed else "FAILED", flush=True)
                        if not passed:
                            raise RuntimeError(f"{name} failed; inspect {logfile}")


if __name__ == "__main__":
    main()
