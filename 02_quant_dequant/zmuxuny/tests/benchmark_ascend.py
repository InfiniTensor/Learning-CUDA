"""Interleaved Ascend baseline/current trials with DSO and output provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HAD = ROOT.parent.name == "03_hadamard_tc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--output", type=Path, default=ROOT / "results/ascend/comparison.json"
    )
    p.add_argument(
        "--large", action="store_true", help="include the 128MiB input cases"
    )
    a = p.parse_args()
    binaries = {"before": a.before.resolve(), "after": a.after.resolve()}
    report = {
        "platform": "Ascend910B2",
        "baseline": "first correct native Ascend C port",
        "trials": a.trials,
        "timing": "ACL events; three warmups per pipeline; complete device stages; exclude allocation, file IO, CPU reference and transfers",
        "binaries": {
            k: {
                "path": str(v),
                "sha256": sha(v),
                "libraries": {f.name: sha(f) for f in v.parent.glob("*.so")},
            }
            for k, v in binaries.items()
        },
        "records": [],
    }
    shapes = []
    for dtype in ([1, 2] if HAD else [0, 1]):
        shapes += [
            (32, 64 if HAD else 1024, dtype),
            (8192 if HAD else 1024, 1024, dtype),
        ]
        if a.large:
            shapes += [(65536 if dtype else 32768, 1024, dtype)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ascend-bench-") as tmp:
        t = Path(tmp)
        for rows, cols, dtype in shapes:
            write_tensor(
                t / "input",
                np.random.default_rng(42).normal(size=(rows, cols)).astype(np.float32),
                dtype,
            )
            repeats = 3 if rows * cols >= 16_000_000 else 10
            for fmt in ["mxfp8", "nvfp4"]:
                record = {
                    "rows": rows,
                    "cols": cols,
                    "dtype": ["fp32", "fp16", "bf16"][dtype],
                    "format": fmt,
                    "input_bytes": rows * cols * (4 if dtype == 0 else 2),
                    "repeats": repeats,
                    "samples": [],
                }
                expected_hash = None
                for trial in range(a.trials):
                    for label in (
                        ["before", "after"] if trial % 2 == 0 else ["after", "before"]
                    ):
                        cmd = [
                            str(binaries[label]),
                            "--input",
                            str(t / "input"),
                            "--output",
                            str(t / "out"),
                            "--packed",
                            str(t / "packed"),
                            "--log",
                            str(t / "metrics"),
                            "--repeats",
                            str(repeats),
                        ]
                        if HAD:
                            cmd += ["--format", fmt, "--materialized_compare", "1"]
                        else:
                            (t / "config").write_text(
                                f'format = "{fmt}"\noutput_type = "fp32"\n'
                            )
                            cmd += ["--config", str(t / "config")]
                        run = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=240
                        )
                        if run.returncode:
                            raise RuntimeError(run.stdout + run.stderr)
                        digest = sha(t / "packed")
                        if expected_hash is None:
                            expected_hash = digest
                        assert digest == expected_hash, (
                            rows,
                            cols,
                            dtype,
                            fmt,
                            label,
                            "packed mismatch",
                        )
                        metrics = json.loads((t / "metrics").read_text())
                        record["samples"].append(
                            {
                                "label": label,
                                "trial": trial,
                                "packed_sha256": digest,
                                "metrics": metrics,
                            }
                        )
                record["median"] = {
                    label: {
                        key: statistics.median(
                            s["metrics"][key]
                            for s in record["samples"]
                            if s["label"] == label
                        )
                        for key, value in record["samples"][0]["metrics"].items()
                        if isinstance(value, (float, int))
                    }
                    for label in binaries
                }
                report["records"].append(record)
                a.output.write_text(json.dumps(report, indent=2) + "\n")
                key = "hadamard_ms" if HAD else "quant_ms"
                print(
                    json.dumps(
                        {k: record[k] for k in ["rows", "cols", "dtype", "format"]}
                    )
                    + f' {record["median"]["before"][key]:.5f} -> {record["median"]["after"][key]:.5f} ms',
                    flush=True,
                )


if __name__ == "__main__":
    main()
