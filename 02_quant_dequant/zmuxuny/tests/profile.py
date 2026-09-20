"""Capture profiler/sanitizer evidence. Unsupported tools are recorded honestly."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]


def capture(cmd, path, timeout=120):
    try:
        p = subprocess.run(
            list(map(str, cmd)), capture_output=True, text=True, timeout=timeout
        )
        output = "$ " + " ".join(map(str, cmd)) + "\n" + p.stdout + p.stderr
        path.write_text("\n".join(line.rstrip() for line in output.splitlines()) + "\n")
        return {"returncode": p.returncode, "log": path.name}
    except (OSError, subprocess.TimeoutExpired) as e:
        path.write_text(str(e) + "\n")
        return {"error": str(e), "log": path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncu",
        default=shutil.which("ncu")
        or "/usr/lib/x86_64-linux-gnu/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu",
    )
    parser.add_argument(
        "--sanitizer", default=shutil.which("compute-sanitizer") or "compute-sanitizer"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lp-profile-") as tmp:
        t = Path(tmp)
        rng = np.random.default_rng(42)
        write_tensor(t / "q.bin", rng.normal(size=(19, 35)), 0)
        cmds = {
            ROOT: [
                ROOT / "build/quantize",
                "--input",
                t / "q.bin",
                "--config",
                ROOT / "configs/nvfp4.toml",
                "--packed",
                t / "q.pack",
                "--output",
                t / "q.out",
                "--log",
                t / "q.json",
                "--repeats",
                "1",
            ]
        }
        for root, cmd in cmds.items():
            status = {}
            for tool in ["memcheck", "racecheck", "synccheck"]:
                status[tool] = capture(
                    [args.sanitizer, "--tool", tool, "--error-exitcode", "99", *cmd],
                    root / f"results/{tool}.txt",
                )
            # A target-only Debian installation has no stock section files.
            # Explicit metrics allow the CLI to reach the device support check.
            status["ncu"] = capture(
                [
                    args.ncu,
                    "--section-folder",
                    t,
                    "--apply-rules",
                    "no",
                    "--metrics",
                    "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "--launch-count",
                    "3",
                    "--target-processes",
                    "all",
                    *cmd,
                ],
                root / "results/ncu.txt",
            )
            (root / "results/profiling.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
