"""Matched complete CPU/GPU jobs including reading, diagnostics, recording and output."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("outputs/end_to_end"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence/end_to_end.json"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(1, 4):
        for backend in (["cuda", "cpu"] if repeat % 2 else ["cpu", "cuda"]):
            label = f"{backend}_{repeat}"
            command = [str(args.exe.resolve()), "--input", str(args.data / "cluster_4096.txt"),
                       "--config", str(args.data / "steps_1000.cfg"), "--backend", backend,
                       "--cpu-threads", str(args.threads), "--output", str(args.out / f"{label}.bin"),
                       "--final-state", str(args.out / f"{label}.csv"), "--log", str(args.out / f"{label}.json")]
            print(f"Matched full job: {label}", flush=True)
            start = time.perf_counter()
            subprocess.run(command, check=True)
            elapsed = time.perf_counter() - start
            entry = json.loads((args.out / f"{label}.json").read_text())
            entry.update(process_wall_seconds=elapsed, command=command, label=label)
            runs.append(entry)
    medians = {backend: statistics.median(x["process_wall_seconds"] for x in runs if x["backend"] == backend)
               for backend in ("cpu", "cuda")}
    report = {"runs": runs, "median_process_seconds": medians, "end_to_end_speedup": medians["cpu"] / medians["cuda"]}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(medians), f"speedup={report['end_to_end_speedup']:.3f}")


if __name__ == "__main__":
    main()
