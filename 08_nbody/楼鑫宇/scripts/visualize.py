#!/usr/bin/env python3
"""Render an XY animation from an N-body result directory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(result_dir: Path):
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    trajectory_path = result_dir / metadata["trajectory"]["file"]
    with trajectory_path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
    if header.size != 2:
        raise ValueError("trajectory header is incomplete")
    particle_count, record_count = map(int, header)
    if particle_count <= 0 or record_count <= 0:
        raise ValueError("trajectory shape must be positive")
    if (particle_count, record_count) != (
        metadata["particle_count"], metadata["record_count"]
    ):
        raise ValueError("trajectory header does not match metadata")
    expected_bytes = 8 + particle_count * record_count * 3 * 4
    if trajectory_path.stat().st_size != expected_bytes:
        raise ValueError("trajectory file size does not match its header")
    trajectory = np.memmap(
        trajectory_path,
        dtype="<f4",
        mode="r",
        offset=8,
        shape=(particle_count, record_count, 3),
    )
    return metadata, trajectory


def axis_limits(values: np.ndarray):
    lower = float(np.min(values))
    upper = float(np.max(values))
    center = 0.5 * (lower + upper)
    half_width = max(0.5 * (upper - lower), 1.0e-3) * 1.08
    return center - half_width, center + half_width


def main():
    parser = argparse.ArgumentParser(description="Render an N-body XY animation")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-particles", type=int, default=4096)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--title", default="N-body gravity simulation")
    args = parser.parse_args()
    if args.max_particles < 1 or args.fps < 1 or args.dpi < 1:
        parser.error("max-particles, fps and dpi must be positive")

    metadata, trajectory = load_trajectory(args.result_dir)
    particle_count, record_count, _ = trajectory.shape
    display_count = min(particle_count, args.max_particles)
    indices = np.linspace(0, particle_count - 1, display_count, dtype=np.int64)
    positions = np.asarray(trajectory[indices], dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError("trajectory contains a non-finite coordinate")

    output = args.output or (args.result_dir / "trajectory_xy.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".mp4":
        raise ValueError("output filename must end in .mp4")

    x_min, x_max = axis_limits(positions[:, :, 0])
    y_min, y_max = axis_limits(positions[:, :, 1])
    span = max(x_max - x_min, y_max - y_min)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor("#070b16")
    ax.set_facecolor("#070b16")
    ax.set_xlim(x_center - 0.5 * span, x_center + 0.5 * span)
    ax.set_ylim(y_center - 0.5 * span, y_center + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.12, linewidth=0.6)
    ax.set_title(args.title)

    colors = np.linspace(0.05, 0.95, display_count)
    marker_size = 45 if display_count <= 16 else max(2.0, 40.0 / np.sqrt(display_count))
    points = ax.scatter(
        positions[:, 0, 0],
        positions[:, 0, 1],
        c=colors,
        cmap="plasma",
        s=marker_size,
        alpha=0.85,
        linewidths=0,
    )
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    trails = []
    if display_count <= 16:
        color_map = plt.get_cmap("plasma")
        for color in colors:
            line, = ax.plot([], [], color=color_map(color), alpha=0.55, linewidth=1.0)
            trails.append(line)

    recorded_steps = metadata["recorded_steps"]
    times = metadata["times"]

    def update(frame):
        points.set_offsets(positions[:, frame, :2])
        step_text.set_text(
            f"step {recorded_steps[frame]}   t = {times[frame]:.4g}   "
            f"shown {display_count}/{particle_count}"
        )
        for particle_index, line in enumerate(trails):
            line.set_data(
                positions[particle_index, : frame + 1, 0],
                positions[particle_index, : frame + 1, 1],
            )
        return [points, step_text, *trails]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=record_count,
        interval=1000 / args.fps,
        blit=True,
    )
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": record_count,
                "fps": args.fps,
                "displayed_particles": display_count,
                "total_particles": particle_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
