"""Compare two actual binaries on identical inputs, interleaving trial order.

Large cases exceed the RTX 4090 D L2 capacity; timings include all GPU steps
reported by each executable. CPU references and file I/O are outside kernel time.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/4090d/extended.json")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--repeats", type=int, default=30)
    a = p.parse_args()
    hadamard = ROOT.parent.name == "03_hadamard_tc"
    shapes = (
        [(65536, 1024, 1), (65536, 1024, 2)]
        if hadamard
        else [(32768, 1024, 0), (65536, 1024, 1)]
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="cuda-large-") as tmp:
        t = Path(tmp)
        for rows, cols, dtype in shapes:
            x = np.random.default_rng(42).normal(size=(rows, cols)).astype(np.float32)
            write_tensor(t / "input", x, dtype)
            del x
            for fmt in ["mxfp8", "nvfp4"]:
                trials = {"before": [], "after": []}
                for i in range(a.trials):
                    for label in (
                        ["before", "after"] if i % 2 == 0 else ["after", "before"]
                    ):
                        binary = getattr(a, label).resolve()
                        cmd = [
                            binary,
                            "--input",
                            t / "input",
                            "--output",
                            t / "output",
                            "--packed",
                            t / "packed",
                            "--log",
                            t / "log",
                            "--repeats",
                            str(a.repeats),
                        ]
                        if hadamard:
                            cmd += ["--format", fmt]
                        else:
                            cmd += ["--config", ROOT / f"configs/{fmt}.toml"]
                        subprocess.run(
                            list(map(str, cmd)), check=True, capture_output=True
                        )
                        trials[label].append(json.loads((t / "log").read_text()))
                        print(rows, cols, dtype, fmt, label, i, flush=True)
                medians = {
                    label: {
                        key: statistics.median(v[key] for v in values)
                        for key, value in values[0].items()
                        if isinstance(value, (int, float))
                    }
                    for label, values in trials.items()
                }
                records.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "dtype": ["fp32", "fp16", "bf16"][dtype],
                        "format": fmt,
                        "repeats": a.repeats,
                        "median": medians,
                        "trials": trials,
                    }
                )
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
