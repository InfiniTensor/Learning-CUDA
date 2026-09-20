"""Run the actual CANN profiler/sanitizer and retain commands and diagnostics."""

import argparse
import csv
import math
import re
import signal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["profile", "sanitize"], required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--sanitizer-tools",
        nargs="+",
        default=["memcheck"],
        choices=["memcheck", "racecheck", "initcheck", "synccheck"],
        help="Only basic memcheck works without source instrumentation.",
    )
    a = p.parse_args()
    out = (a.output or ROOT / "results/ascend" / a.tool).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build" / "ascend"
    binary = build / ("hadamard" if HADAMARD else "quantize")
    home = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    records = []
    # Tails exercise Cube padding, odd quantized rows and non-aligned DMA.
    shapes = [(19, 64), (33, 1024)] if HADAMARD else [(7, 35), (65, 1024)]
    for fmt in ["mxfp8", "nvfp4"]:
        for dtype in ([1, 2] if HADAMARD else [0, 1]):
            rows, cols = shapes[dtype % 2]
            # Basic memcheck serializes scalar accesses; a short 1024-wide tail
            # exercises the same DMA/UB boundaries within a bounded tool run.
            if a.tool == "sanitize" and cols == 1024:
                if HADAMARD:
                    rows = 3
                else:
                    # Odd packed tails plus >48 decode tiles exercise grid stride.
                    rows, cols = 17, 1025
            case = out / f"{fmt}_{['fp32','fp16','bf16'][dtype]}"
            case.mkdir(exist_ok=True)
            write_tensor(
                case / "input",
                np.random.default_rng(42).normal(size=(rows, cols)).astype(np.float32),
                dtype,
            )
            cmd = [
                str(binary),
                "--input",
                str(case / "input"),
                "--output",
                str(case / "out"),
                "--packed",
                str(case / "packed"),
                "--log",
                str(case / "metrics.json"),
                "--repeats",
                "2",
            ]
            if HADAMARD:
                cmd += [
                    "--format",
                    fmt,
                    "--materialized_compare",
                    "1",
                    "--random_sign",
                    "1",
                ]
            else:
                (case / "config").write_text(
                    f'format = "{fmt}"\noutput_type = "fp32"\n'
                )
                cmd += ["--config", str(case / "config")]
            tools = a.sanitizer_tools if a.tool == "sanitize" else ["msprof"]
            for tool in tools:
                if a.tool == "sanitize":
                    full = [
                        str(home / "tools/mssanitizer/bin/mssanitizer"),
                        "--tool=" + tool,
                        "--",
                        *cmd,
                    ]
                else:
                    full = [
                        "msprof",
                        "--output=" + str(case / "trace"),
                        "--aic-metrics=PipeUtilization",
                        "--task-time=l2",
                        *cmd,
                    ]
                log = case / f"{tool}.log"
                try:
                    with log.open("w") as f:
                        run = subprocess.Popen(
                            full,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            rc = run.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            os.killpg(run.pid, signal.SIGTERM)
                            try:
                                run.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(run.pid, signal.SIGKILL)
                                run.wait()
                            raise
                except subprocess.TimeoutExpired:
                    rc = "TIMEOUT"
                text = log.read_text(errors="replace")
                record = {
                    "case": case.name,
                    "tool": tool,
                    "command": full,
                    "returncode": rc,
                    "log": str(log.relative_to(out)),
                }
                record["status"] = "FAILED"
                if a.tool == "sanitize":
                    starts = len(re.findall(r"Start .* sanitizer on kernel", text))
                    finishes = len(
                        re.findall(
                            r"Sanitizer finished on kernel .* No error detected", text
                        )
                    )
                    warning = bool(
                        re.search(
                            r"(?im)^\[mssanitizer\]\s+(ERROR|WARN|Warning|Error)", text
                        )
                    )
                    record.update(
                        mode="basic binary checks; no source-level sanitizer instrumentation",
                        kernels_started=starts,
                        kernels_clean=finishes,
                        warnings=warning,
                    )
                    if (
                        "is not supported on the kernel that without" in text
                        and starts == 0
                    ):
                        record["status"] = "UNSUPPORTED_NO_INSTRUMENTATION"
                    elif rc == 0 and starts > 0 and starts == finishes and not warning:
                        record["status"] = "PASS_BASIC"
                else:
                    rows_data = []
                    for csv_file in (case / "trace").rglob("op_summary*.csv"):
                        with csv_file.open() as f:
                            rows_data.extend(csv.DictReader(f))
                    valid = [
                        r
                        for r in rows_data
                        if math.isfinite(float(r["Task Duration(us)"]))
                        and float(r["Task Duration(us)"]) > 0
                    ]
                    record.update(
                        kernel_records=len(rows_data), valid_durations=len(valid)
                    )
                    if rc == 0 and valid and len(valid) == len(rows_data):
                        record["status"] = "PASS_PROFILE"
                records.append(record)
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "binaries": {
                                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in build.iterdir()
                                if f.is_file()
                            },
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(record), flush=True)
                if rc == "TIMEOUT":
                    raise SystemExit("tool timed out; stop further checks")


if __name__ == "__main__":
    main()
