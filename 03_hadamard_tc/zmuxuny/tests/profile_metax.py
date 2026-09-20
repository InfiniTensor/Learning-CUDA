"""Capture C500 device traces separately from uninstrumented event benchmarks."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from generate import generate
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"
MACA_PATH = Path(os.environ.get("MACA_PATH", "/opt/maca"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path,
                   default=ROOT / "build/metax" / ("hadamard" if HADAMARD else "quantize"))
    p.add_argument("--output", type=Path, default=ROOT / "results/c500/profile")
    p.add_argument("--tracer", default=str(MACA_PATH / "bin/mcTracer"))
    args = p.parse_args()
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(MACA_PATH / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="metax-profile-") as tmp:
        t = Path(tmp)
        rows, cols = (8192, 128) if HADAMARD else (1024, 1024)
        for dtype in ([1, 2] if HADAMARD else [1]):
            write_tensor(t / "input", generate(rows, cols, "normal"), dtype)
            for fmt in ["mxfp8", "nvfp4"]:
                label = f"{fmt}_{['fp32', 'fp16', 'bf16'][dtype]}"
                dest = (args.output / label).resolve()
                dest.mkdir(parents=True, exist_ok=True)
                command = [str(binary), "--input", str(t / "input"),
                           "--output", str(t / "output"), "--packed", str(t / "packed"),
                           "--log", str(dest / "instrumented_metrics.json"), "--repeats", "20"]
                if HADAMARD:
                    command += ["--format", fmt, "--materialized_compare", "1"]
                else:
                    command += ["--config", str(ROOT / "configs" / f"{fmt}.toml")]
                proc = subprocess.run([args.tracer, "--odname", label, *command],
                                      capture_output=True, text=True, env=env, cwd=args.output.resolve())
                (dest / "tracer.log").write_text(proc.stdout + proc.stderr)
                if proc.returncode:
                    raise RuntimeError(proc.stdout + proc.stderr)
                traces = sorted(dest.glob("tracer_out-*.json"), key=lambda f: f.stat().st_mtime)
                if not traces:
                    raise RuntimeError("mcTracer produced no trace: " + proc.stdout + proc.stderr)
                trace = traces[-1]
                kernels = collections.defaultdict(list)
                for event in json.loads(trace.read_text())["traceEvents"]:
                    if event.get("ph") == "X" and "grid" in event.get("args", {}):
                        kernels[event["name"]].append(event)
                for name, events in kernels.items():
                    durations = [e["dur"] / 1000 for e in events]  # mcTracer timestamps are ns
                    records.append({"case": label, "kernel": name, "count": len(events),
                                    "mean_us": statistics.mean(durations),
                                    "median_us": statistics.median(durations),
                                    "min_us": min(durations), "max_us": max(durations),
                                    "launch": events[0]["args"],
                                    "trace": str(trace.relative_to(args.output.resolve()))})
                print(label, "kernels", sum(len(v) for v in kernels.values()), flush=True)
    report = {"tool": "MACA mcTracer", "gpu": "MetaX C500", "rows": rows, "cols": cols,
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "timing_note": "Instrumented trace durations; use benchmark JSON for performance comparisons.",
              "kernels": records}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
