"""Native-Linux GPU evidence: three sanitizers, NCU, Nsight Systems and SASS.

Run after building for the current GPU. Profiling is separate from benchmarks.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"


def capture(command, output, timeout=120):
    command = list(map(str, command))
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        raw = "$ " + " ".join(command) + "\n" + p.stdout + p.stderr
        output.write_text(
            "\n".join(line.rstrip() for line in raw.splitlines()).rstrip() + "\n"
        )
        return {"returncode": p.returncode, "log": output.name}
    except (OSError, subprocess.TimeoutExpired) as e:
        output.write_text(str(e) + "\n")
        return {"error": str(e), "log": output.name}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "results/4090d/after")
    p.add_argument(
        "--sanitizer",
        default=shutil.which("compute-sanitizer")
        or "/usr/local/cuda/compute-sanitizer/compute-sanitizer",
    )
    p.add_argument(
        "--binary",
        type=Path,
        default=ROOT / ("build/hadamard" if HADAMARD else "build/quantize"),
    )
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    with tempfile.TemporaryDirectory(prefix="cuda-profile-") as tmp:
        t = Path(tmp)

        def command(rows, cols, dtype, fmt, repeats=1):
            write_tensor(
                t / "input.bin",
                np.random.default_rng(42).normal(size=(rows, cols)),
                dtype,
            )
            cmd = [
                args.binary.resolve(),
                "--input",
                t / "input.bin",
                "--output",
                t / "output.bin",
                "--packed",
                t / "packed.bin",
                "--log",
                t / "metrics.json",
                "--repeats",
                str(repeats),
            ]
            if HADAMARD:
                cmd += ["--format", fmt, "--random_sign", "1"]
            else:
                cmd += ["--config", ROOT / f"configs/{fmt}.toml"]
            return cmd

        shapes = (
            [(19, 128, 1), (35, 1024, 1), (35, 1024, 2)]
            if HADAMARD
            else [(19, 35, 0), (1031, 1025, 1), (35, 1024, 0), (35, 1024, 1)]
        )
        for rows, cols, dtype in shapes:
            for fmt in ["mxfp8", "nvfp4"]:
                cmd = command(rows, cols, dtype, fmt)
                if HADAMARD:
                    cmd += ["--materialized_compare", "1"]
                elif rows == 35 and dtype == 1:
                    (t / "config").write_text(
                        f'format = "{fmt}"\noutput_type = "bf16"\n'
                    )
                    cmd[-1] = t / "config"
                for tool in ["memcheck", "racecheck", "synccheck"]:
                    key = f"{tool}_{rows}x{cols}_dtype{dtype}_{fmt}"
                    status[key] = capture(
                        [
                            args.sanitizer,
                            "--tool",
                            tool,
                            "--error-exitcode",
                            "99",
                            *cmd,
                        ],
                        out / f"{key}.txt",
                    )
                    print(key, status[key], flush=True)
        cmd = command(
            8192 if HADAMARD else 1024, 128 if HADAMARD else 1024, 1, "nvfp4", 10
        )
        status["ncu"] = capture(
            ["ncu", "--set", "basic", "--launch-count", "3", *cmd], out / "ncu.txt"
        )
        for fmt in ["mxfp8", "nvfp4"]:
            cmd = command(
                8192 if HADAMARD else 1024, 128 if HADAMARD else 1024, 1, fmt, 20
            )
            status[f"nsys_{fmt}"] = capture(
                [
                    "nsys",
                    "profile",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--trace=cuda,nvtx",
                    "--force-overwrite=true",
                    "-o",
                    out / f"timeline_{fmt}",
                    *cmd,
                ],
                out / f"nsys_{fmt}.txt",
            )
            if status[f"nsys_{fmt}"].get("returncode") == 0:
                status[f"nsys_stats_{fmt}"] = capture(
                    [
                        "nsys",
                        "stats",
                        "--force-export=true",
                        "--report",
                        "cuda_gpu_kern_sum,cuda_api_sum",
                        "--format",
                        "csv",
                        out / f"timeline_{fmt}.nsys-rep",
                    ],
                    out / f"nsys_stats_{fmt}.csv",
                )
        sass = subprocess.run(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", args.binary.resolve()],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        groups = {"conversion_instructions.txt": []}
        if HADAMARD:
            groups["tensor_core_instructions.txt"] = []
        function = ""
        for line in sass.splitlines():
            if "Function :" in line:
                function = line.strip()
            if "E4M3" in line or "BF16.F32" in line:
                groups["conversion_instructions.txt"] += [function, line.rstrip()]
            if HADAMARD and "HMMA." in line:
                groups["tensor_core_instructions.txt"] += [function, line.rstrip()]
        for filename, lines in groups.items():
            (out / filename).write_text("\n".join(lines) + "\n")
        status["resources"] = capture(
            [
                "/usr/local/cuda/bin/cuobjdump",
                "--dump-resource-usage",
                args.binary.resolve(),
            ],
            out / "resources.txt",
        )
        status["environment"] = capture(["nvidia-smi", "-q"], out / "nvidia-smi.txt")
    (out / "profiling.json").write_text(json.dumps(status, indent=2) + "\n")
    failures = [
        k
        for k, v in status.items()
        if k.startswith(("memcheck", "racecheck", "synccheck"))
        and v.get("returncode") != 0
    ]
    if failures:
        raise SystemExit("Sanitizer failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
