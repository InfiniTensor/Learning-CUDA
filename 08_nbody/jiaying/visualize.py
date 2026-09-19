#!/usr/bin/env python3
"""Animate N-body trajectories in 2D/3D, with optional proximity highlighting."""
import argparse
from pathlib import Path
import numpy as np
from scripts.trajectory import load_trajectory, close_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--save", type=Path, help=".gif (Pillow), .mp4 (ffmpeg), or .html (embedded JS)")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-particles", type=int, default=6000)
    parser.add_argument("--trail", type=int, default=40)
    parser.add_argument("--risk-radius", type=float, default=0)
    parser.add_argument("--title", default="N-BODY / GRAVITATIONAL DYNAMICS")
    args = parser.parse_args()
    if min(args.fps, args.stride, args.max_particles, args.trail) < 1 or not np.isfinite(args.risk_radius) or args.risk_radius < 0:
        parser.error("Counts must be positive; risk-radius must be finite and nonnegative")
    import matplotlib
    if args.save or args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    points, times, metadata = load_trajectory(args.trajectory)
    n, records, _ = points.shape
    ids = np.linspace(0, n - 1, min(n, args.max_particles), dtype=int)
    frames = list(range(0, records, args.stride))
    if frames[-1] != records - 1:
        frames.append(records - 1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#dce8f5", "axes.labelcolor": "#93abc3", "xtick.color": "#8298b0", "ytick.color": "#8298b0"})
    fig = plt.figure(figsize=(9.6, 7.2), facecolor="#08121e")
    ax = fig.add_subplot(111, projection="3d" if args.dimensions == 3 else None, facecolor="#08121e")
    fig.subplots_adjust(left=0.1, right=0.93, bottom=0.12, top=0.84)
    fig.text(0.08, 0.935, args.title, fontsize=16, weight="bold")
    fig.text(0.08, 0.89, f"CUDA SIMULATION     /     {n:,} BODIES     /     {records:,} RECORDED FRAMES", color="#779bbd", fontsize=10)
    caption = fig.text(0.1, 0.045, "", color="#84d7eb")
    # Bound all displayed trajectories, including outliers; no invisible clipping.
    limits = np.zeros((2, 3))
    limits[0] = np.inf; limits[1] = -np.inf
    for k in frames:
        p = points[ids, k, :]
        if not np.all(np.isfinite(p)):
            raise ValueError("Trajectory contains non-finite positions")
        limits[0] = np.minimum(limits[0], p.min(axis=0))
        limits[1] = np.maximum(limits[1], p.max(axis=0))
    center = limits.mean(axis=0)
    radius = max(float(np.max(limits[1] - limits[0])) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_xlabel("x / normalized distance"); ax.set_ylabel("y / normalized distance")
    if args.dimensions == 3:
        ax.set_zlim(center[2] - radius, center[2] + radius); ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.03, 0.07, 0.12, 1))
    else:
        ax.set_aspect("equal")
        ax.grid(color="#203145", alpha=0.6, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#203145")
    initial_radius = np.linalg.norm(points[ids, 0], axis=1)
    colors = plt.colormaps["cool"](Normalize(0, max(float(initial_radius.max()), 1e-9))(initial_radius))
    if n <= 12:
        colors = plt.colormaps["cool"](np.linspace(0.1, 0.9, len(ids)))
    p = points[ids, 0]
    scatter = ax.scatter(*[p[:, k] for k in range(args.dimensions)], s=42 if n <= 8 else 4, c=colors, alpha=0.8)
    trail_ids = ids[:min(12, len(ids))]
    lines = [ax.plot(*([[]] * args.dimensions), color=colors[k], alpha=0.5, linewidth=0.8)[0] for k in range(len(trail_ids))]

    def update(frame):
        p = points[ids, frame]
        if args.dimensions == 3:
            scatter._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        else:
            scatter.set_offsets(p[:, :2])
        risk_text = ""
        if args.risk_radius:
            marked, pairs = close_pairs(points[:, frame], args.risk_radius)
            current = colors.copy(); current[marked[ids]] = (1, 0.3, 0.22, 1)
            scatter.set_facecolor(current)
            risk_text = f"   |   CLOSE PAIRS {pairs:,}  (< {args.risk_radius:g}, 3D)"
        for line, particle in zip(lines, trail_ids):
            track = points[particle, max(0, frame - args.trail + 1):frame + 1]
            line.set_data(track[:, 0], track[:, 1])
            if args.dimensions == 3:
                line.set_3d_properties(track[:, 2])
        unit = "record" if metadata.get("time_is_record_index") else "t"
        caption.set_text(f"{unit} = {times[frame]:.3f}   |   FRAME {frame + 1:04d} / {records:04d}{risk_text}")
        return scatter, caption, *lines

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False, cache_frame_data=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.save.suffix.lower()
        if suffix == ".gif":
            animation.save(args.save, writer=PillowWriter(fps=args.fps), dpi=90)
        elif suffix == ".html":
            args.save.write_text(animation.to_jshtml(fps=args.fps), encoding="utf-8")
        elif suffix == ".mp4":
            animation.save(args.save, writer="ffmpeg", fps=args.fps, dpi=120)
        else:
            parser.error("--save must end in .gif, .mp4 or .html")
        print(f"Saved animation: {args.save}")
    if args.snapshot:
        update(records - 1)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.snapshot, dpi=140, facecolor=fig.get_facecolor())
    if not args.save and not args.snapshot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
