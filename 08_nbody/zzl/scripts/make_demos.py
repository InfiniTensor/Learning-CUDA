"""Generate real simulation data and rendered demo animations (no external assets)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate import make_particles, write_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "demos")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for scene, n, dt, steps, interval, eps in (
        ("binary", 2, 0.003, 3000, 25, 0.001),
        ("cluster", 4096, 0.0015, 3000, 25, 0.02),
        ("orbits", 128, 0.003, 4000, 40, 0.02),
        ("orbits-control", 128, 0.003, 4000, 40, 0.02),
        ("encounter", 96, 0.002, 1600, 16, 0.01),
    ):
        print(f"Simulating {scene}", flush=True)
        data = make_particles(scene, n, softening=eps)
        np.savetxt(output / f"{scene}.txt", data, fmt="%.9g", header="x y z vx vy vz mass")
        write_config(output / f"{scene}.cfg", dt, steps, interval, eps)
        command = [str(args.exe.resolve()), "--input", str(output / f"{scene}.txt"), "--config", str(output / f"{scene}.cfg"),
                   "--output", str(output / f"{scene}.bin"), "--log", str(output / f"{scene}.json"),
                   "--final-state", str(output / f"{scene}_final.csv")]
        subprocess.run(command, check=True)
        records.append(json.loads((output / f"{scene}.json").read_text()))
    for scene, dimension, title, extra in (
        ("binary", 2, "TWO BODIES / ONE ORBIT", ["--trails", "2", "--trail-length", "100"]),
        ("cluster", 3, "STAR CLUSTER / SELF-GRAVITY", ["--max-particles", "1800", "--trails", "18"]),
        ("orbits", 2, "ORBITAL PERTURBATION / CONTROL", ["--reference", str(output / "orbits-control.bin"), "--trails", "14"]),
        ("encounter", 3, "CLOSE ENCOUNTERS / PROXIMITY", ["--risk-distance", "0.035", "--risk-csv", str(output / "encounters.csv"),
                                                          "--snapshot-frame", "50", "--trails", "12"]),
    ):
        print(f"Rendering {scene}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "visualize.py"), str(output / f"{scene}.bin"),
                        "--dimension", str(dimension), "--title", title, "--fps", "25",
                        "--save", str(output / f"{scene}.gif"), "--snapshot", str(output / f"{scene}.png"), *extra], check=True)
    (output / "demo_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
