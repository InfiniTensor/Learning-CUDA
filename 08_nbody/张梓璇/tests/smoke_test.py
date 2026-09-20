#!/usr/bin/env python3
"""分别跑各个 kernel，并检查轨迹头、文件长度、性能日志和 CPU 误差。"""

import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path


def parse_log(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key] = value
    return values


def read_positions(path: Path) -> tuple[int, int, array]:
    with path.open("rb") as file:
        particle_count, record_count = struct.unpack("=ii", file.read(8))
        values = array("f")
        values.fromfile(file, particle_count * record_count * 3)
    return particle_count, record_count, values


def run_case(
    executable: Path, project: Path, kernel: str, config_name: str, label: str, temp: Path,
    cuda_graph: bool = False,
) -> None:
    trajectory = temp / f"{label}.bin"
    log = temp / f"{label}.log"
    command = [
            str(executable),
            str(project / "tests" / "small_test.txt"),
            str(project / "tests" / config_name),
            "--kernel", kernel,
            "--output", str(trajectory),
            "--performance-log", str(log),
            "--cpu-validation", "on",
        ]
    if cuda_graph:
        command.append("--cuda-graph")
    subprocess.run(command, check=True)
    with trajectory.open("rb") as file:
        particle_count, record_count = struct.unpack("=ii", file.read(8))
    assert particle_count == 4
    assert record_count == 5
    assert trajectory.stat().st_size == 8 + particle_count * record_count * 3 * 4

    values = parse_log(log)
    required = {
        "gpu_simulation_ms", "cpu_simulation_ms", "cpu_gpu_speedup",
        "gpu_allocated_bytes", "initial_energy", "final_energy",
        "relative_energy_error", "max_position_error", "max_velocity_error",
    }
    assert required.issubset(values), required - values.keys()
    assert float(values["max_position_error"]) < 1.0e-4
    assert float(values["max_velocity_error"]) < 1.0e-4
    assert float(values["relative_energy_error"]) < 1.0e-3


def compare_trajectories(first: Path, second: Path, tolerance: float = 1.0e-5) -> None:
    first_n, first_records, first_values = read_positions(first)
    second_n, second_records, second_values = read_positions(second)
    assert (first_n, first_records) == (second_n, second_records)
    max_difference = max(abs(a - b) for a, b in zip(first_values, second_values))
    assert max_difference < tolerance, max_difference


def check_two_body_orbit(executable: Path, project: Path, temp: Path) -> None:
    trajectory = temp / "two_body.bin"
    log = temp / "two_body.log"
    subprocess.run(
        [
            str(executable), str(project / "tests" / "two_body.txt"),
            str(project / "tests" / "two_body.cfg"),
            "--kernel", "tiled-float4", "--output", str(trajectory),
            "--performance-log", str(log), "--cpu-validation", "on",
        ],
        check=True,
    )
    particle_count, _, values = read_positions(trajectory)
    assert particle_count == 2
    distances = []
    for offset in range(0, len(values), particle_count * 3):
        dx = values[offset + 3] - values[offset]
        dy = values[offset + 4] - values[offset + 1]
        dz = values[offset + 5] - values[offset + 2]
        distances.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    assert max(abs(distance - 1.0) for distance in distances) < 1.0e-3


def main() -> None:
    executable = Path(sys.argv[1]).resolve()
    project = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="nbody_smoke_") as directory:
        temp = Path(directory)
        run_case(executable, project, "baseline", "small_test.cfg", "baseline_leapfrog", temp)
        run_case(executable, project, "tiled", "small_test.cfg", "tiled_leapfrog", temp)
        run_case(executable, project, "tiled-float4", "small_test.cfg", "float4_leapfrog", temp)
        run_case(executable, project, "split-source", "small_test.cfg", "split_source", temp)
        run_case(executable, project, "split-source", "small_test.cfg", "split_graph", temp, True)
        run_case(executable, project, "tiled", "small_euler.cfg", "tiled_euler", temp)
        compare_trajectories(temp / "baseline_leapfrog.bin", temp / "tiled_leapfrog.bin")
        compare_trajectories(temp / "baseline_leapfrog.bin", temp / "float4_leapfrog.bin")
        compare_trajectories(temp / "baseline_leapfrog.bin", temp / "split_source.bin")
        compare_trajectories(temp / "baseline_leapfrog.bin", temp / "split_graph.bin")
        check_two_body_orbit(executable, project, temp)
    print("four kernels, CUDA Graph, two integrators and two-body orbit tests passed")


if __name__ == "__main__":
    main()
