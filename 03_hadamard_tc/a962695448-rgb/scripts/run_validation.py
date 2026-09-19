#!/usr/bin/env python3
"""Reproduce correctness, rejected-CLI, and optional measured benchmark logs."""
import argparse
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--label", default="local")
    args = parser.parse_args()
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label):
        parser.error("--label must contain only letters, digits, '-' or '_'")
    root = Path(__file__).resolve().parents[1]
    executable = root / "build/hadamard"
    results = root / "results"
    results.mkdir(exist_ok=True)
    with (results / f"validation_{args.label}.log").open("w") as log:
        def run(arguments, expected=0):
            command = [str(executable), *map(str, arguments)]
            log.write("COMMAND " + repr(command) + "\n")
            log.flush()
            start = time.monotonic()
            result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=180)
            log.write(result.stdout)
            log.write(f"EXIT_CODE {result.returncode}; ELAPSED_SECONDS {time.monotonic()-start:.3f}\n\n")
            log.flush()
            if result.returncode != expected:
                raise RuntimeError(f"unexpected exit {result.returncode}, expected {expected}: {command}\n{result.stdout}")
            print("PASS", " ".join(map(str, arguments)), flush=True)

        run(["--self-test"])
        for invalid in (
            ["--benchmark", "--dim", "0"],
            ["--benchmark", "--dim", "3"],
            ["--benchmark", "--dim", "512"],
            ["--benchmark", "--batch", "-1"],
            ["--benchmark", "--seq", "0"],
            ["--benchmark", "--heads", "abc"],
            ["--benchmark", "--dtype", "fp32"],
            ["--benchmark", "--scale", "nan"],
            ["--benchmark", "--scale", "2"],
            ["--benchmark", "--batch", "18446744073709551615", "--seq", "2"],
            ["--benchmark", "--batch", "18446744073709551616"],
            ["--benchmark", "--repetitions", "0"],
            ["--benchmark", "--warmup", "-1"],
            ["--benchmark", "--dim"],
            ["--unsupported"],
        ):
            run(invalid, expected=2)
        if args.benchmark:
            csv = results / f"benchmark_{args.label}.csv"
            # Do not mix independent runs in one CSV silently.
            if csv.exists():
                raise FileExistsError(f"Choose a fresh --label; CSV exists: {csv}")
            shapes = [(1, 1, 1, 1), (1, 1, 1, 16), (1, 1, 17, 256),
                      (4, 128, 8, 16), (4, 128, 8, 64), (4, 128, 8, 256),
                      (4, 512, 8, 256)]
            for dtype in ("fp16", "bf16"):
                for batch, seq, heads, dim in shapes:
                    run(["--benchmark", "--batch", batch, "--seq", seq, "--heads", heads,
                         "--dim", dim, "--dtype", dtype, "--repetitions", 300,
                         "--warmup", 30, "--csv", csv])
                run(["--benchmark", "--batch", 1, "--seq", 257, "--heads", 1,
                     "--dim", 256, "--dtype", dtype, "--normalize", "--repetitions", 300,
                     "--warmup", 30, "--csv", csv])


if __name__ == "__main__":
    main()
