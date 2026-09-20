"""Animate CUDA N-body trajectories in 2D/3D using Matplotlib FuncAnimation."""
import argparse
import csv
from pathlib import Path
import numpy as np
from trajectory import close_pairs, load_trajectory


def create_animation(path, dimensions=2, max_particles=2000, trails=12, trail_length=35,
                     fps=25, risk_distance=None, max_pairs=2000, reference=None,
                     title="N-BODY / GRAVITY LAB", frame_stride=1):
    import matplotlib.pyplot as plt
    from matplotlib import animation, colormaps

    positions, metadata = load_trajectory(path)
    particles, records, _ = positions.shape
    ids = np.linspace(0, particles - 1, min(particles, max_particles), dtype=int)
    shown = np.asarray(positions[ids])
    times = np.asarray(metadata["steps"]) * metadata["dt"]
    comparison = None
    rms = None
    if reference:
        comparison, other = load_trajectory(reference)
        if comparison.shape != positions.shape or other["steps"] != metadata["steps"] or not np.isclose(other["dt"], metadata["dt"]):
            raise ValueError("Reference trajectory must have identical particle IDs and saved times")
        # Quantitative displacement is for every particle, not just the displayed sample.
        squared = np.zeros(records)
        for base in range(0, particles, 256):
            delta = np.asarray(positions[base:base + 256], dtype=float) - comparison[base:base + 256]
            squared += np.sum(delta * delta, axis=(0, 2))
        rms = np.sqrt(squared / particles)

    background = "#080f1d"
    foreground = "#e0e9f5"
    muted = "#8e9db3"
    cyan = "#67e3e0"
    orange = "#ffad66"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(12.8, 7.6), facecolor=background)
    grid = fig.add_gridspec(3, 4, left=0.055, right=0.965, top=0.83, bottom=0.12,
                           wspace=0.6, hspace=0.65)
    ax = fig.add_subplot(grid[:, :3], projection="3d" if dimensions == 3 else None)
    ax.set_facecolor(background)
    fig.text(0.055, 0.945, title, color=foreground, fontsize=22, weight="bold")
    fig.text(0.055, 0.895, "Newtonian gravity  /  softened point masses  /  normalized units", color=muted)
    time_label = fig.text(0.965, 0.895, "", color=cyan, ha="right", family="monospace")
    fig.text(0.055, 0.045, f"{particles:,} particles  |  {len(ids):,} displayed  |  {records} saved frames", color=muted, fontsize=9)
    center = (shown.max(axis=(0, 1)) + shown.min(axis=(0, 1))) / 2
    half = max(float(np.ptp(shown, axis=(0, 1))[:dimensions].max()) * 0.54, 0.1)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xlabel("x", color=muted)
    ax.set_ylabel("y", color=muted)
    if dimensions == 3:
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_zlabel("z", color=muted)
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
        ax.view_init(elev=24, azim=-62)
    else:
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color("#273448")
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(alpha=0.12, color=muted)
    radius0 = np.linalg.norm(shown[:, 0], axis=1)
    base_colors = colormaps["cool"](radius0 / max(float(radius0.max()), 1e-9))
    base_colors[:, 3] = 0.82
    dot_size = max(5, min(50, 6000 / len(ids)))
    if dimensions == 3:
        dots = ax.scatter(*shown[:, 0].T, s=dot_size, c=base_colors, depthshade=False)
    else:
        dots = ax.scatter(shown[:, 0, 0], shown[:, 0, 1], s=dot_size, c=base_colors, linewidths=0)
    line_ids = np.linspace(0, len(ids) - 1, min(trails, len(ids)), dtype=int)
    lines = [ax.plot([], [], *([[]] if dimensions == 3 else []), color=cyan, alpha=0.45, lw=0.8)[0] for _ in line_ids]
    reference_dots = None
    if comparison is not None:
        if dimensions == 3:
            reference_dots = ax.scatter(*comparison[ids, 0].T, s=dot_size, facecolors="none", edgecolors=muted, alpha=0.4)
        else:
            reference_dots = ax.scatter(comparison[ids, 0, 0], comparison[ids, 0, 1], s=dot_size,
                                        facecolors="none", edgecolors=muted, alpha=0.4, linewidths=0.6)

    panel = fig.add_subplot(grid[0, 3])
    panel.axis("off")
    panel.text(0, 1, "SIMULATION", color=cyan, fontsize=10, weight="bold", transform=panel.transAxes)
    panel.text(0, 0.75, f"{dimensions}D trajectory view\n\nBinary float32 data\n\nInitial + final frames", color=foreground,
               fontsize=10, va="top", transform=panel.transAxes)
    chart = fig.add_subplot(grid[1, 3])
    chart.set_facecolor(background)
    if rms is not None:
        values, label = rms, "RMS DISPLACEMENT"
    else:
        values = np.median(np.linalg.norm(shown, axis=2), axis=0)
        label = "MEDIAN RADIUS (shown)"
    chart.plot(times, values, color=muted, alpha=0.5, lw=1)
    current, = chart.plot([], [], color=cyan, lw=2)
    chart.set_title(label, loc="left", fontsize=9, color=foreground, pad=12)
    chart.set_xlabel("time" if not metadata.get("time_is_record_index") else "record", color=muted, fontsize=8)
    chart.tick_params(colors=muted, labelsize=7)
    for spine in chart.spines.values():
        spine.set_color("#273448")
    risk_panel = fig.add_subplot(grid[2, 3])
    risk_panel.axis("off")
    risk_panel.text(0, 1, "PROXIMITY SCREEN" if risk_distance else "READOUT", color=orange if risk_distance else cyan,
                    fontsize=9, weight="bold", transform=risk_panel.transAxes)
    risk_label = risk_panel.text(0, 0.74, "", color=foreground, va="top", fontsize=10, transform=risk_panel.transAxes)
    risk_panel.text(0, 0, "Saved-frame distances only.\nNo collision dynamics." if risk_distance else
                    ("Hollow dots: reference run\nMatched particle IDs." if reference else "Traces show recent history.\nAxes remain fixed."),
                    color=muted, fontsize=8, va="bottom", transform=risk_panel.transAxes)
    risk_cache = {}

    def update(frame):
        points = shown[:, frame]
        if dimensions == 3:
            dots._offsets3d = tuple(points.T)
        else:
            dots.set_offsets(points[:, :2])
        colors = base_colors.copy()
        if risk_distance:
            if frame not in risk_cache:
                risk_cache[frame] = close_pairs(positions[:, frame], risk_distance, max_pairs)
            pairs, truncated = risk_cache[frame]
            flagged = {i for a, b, _ in pairs for i in (a, b)}
            colors[np.isin(ids, list(flagged))] = (1.0, 0.55, 0.27, 1.0)
            risk_label.set_text(f"d < {risk_distance:g}\n\n{len(pairs):,}{'+' if truncated else ''} close pairs\n" +
                                ("Pair cap reached" if truncated else "All particles screened"))
        else:
            risk_label.set_text(f"{'RMS shift' if rms is not None else 'Median radius'}\n\n{values[frame]:.4f}")
        dots.set_facecolors(colors)
        for line, i in zip(lines, line_ids):
            trace = shown[i, max(0, frame - trail_length + 1):frame + 1]
            if dimensions == 3:
                line.set_data_3d(*trace.T)
            else:
                line.set_data(trace[:, 0], trace[:, 1])
        if reference_dots is not None:
            ref_points = comparison[ids, frame]
            if dimensions == 3:
                reference_dots._offsets3d = tuple(ref_points.T)
            else:
                reference_dots.set_offsets(ref_points[:, :2])
        current.set_data(times[:frame + 1], values[:frame + 1])
        time_label.set_text(f"t = {times[frame]:8.3f}    frame {frame + 1:04d}/{records:04d}")
        return dots, time_label, current, risk_label, *lines

    frames = list(range(0, records, frame_stride))
    if frames[-1] != records - 1:
        frames.append(records - 1)
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                                   blit=False, repeat=True, cache_frame_data=False)
    update(0)
    return fig, anim, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2)
    parser.add_argument("--max-particles", type=int, default=2000)
    parser.add_argument("--trails", type=int, default=12)
    parser.add_argument("--trail-length", type=int, default=35)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--risk-distance", type=float)
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--risk-csv", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--title", default="N-BODY / GRAVITY LAB")
    parser.add_argument("--save", type=Path, help="GIF, MP4 (FFmpeg required), or standalone HTML")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-frame", type=int, default=-1)
    args = parser.parse_args()
    if min(args.max_particles, args.trail_length, args.fps, args.frame_stride, args.max_pairs) <= 0 or args.trails < 0:
        parser.error("Positive counts required; trails may be zero")
    if args.risk_distance is not None and (not np.isfinite(args.risk_distance) or args.risk_distance <= 0):
        parser.error("risk-distance must be positive and finite")
    if args.risk_csv and not args.risk_distance:
        parser.error("--risk-csv requires --risk-distance")
    destinations = [x.resolve() for x in (args.save, args.snapshot, args.risk_csv) if x]
    if len(set(destinations)) != len(destinations) or any(x.exists() for x in destinations):
        parser.error("Choose distinct new output paths; existing files are preserved")
    if args.save and args.save.suffix.lower() not in {".gif", ".mp4", ".html"}:
        parser.error("Animation output must be .gif, .mp4 or .html")
    if args.save or args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig, anim, update = create_animation(args.trajectory, args.dimension, args.max_particles,
                                        args.trails, args.trail_length, args.fps, args.risk_distance,
                                        args.max_pairs, args.reference, args.title, args.frame_stride)
    positions, metadata = load_trajectory(args.trajectory)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if args.risk_csv:
        with args.risk_csv.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "step", "time", "particle_i", "particle_j", "distance", "truncated"])
            for frame, step in enumerate(metadata["steps"]):
                pairs, truncated = close_pairs(positions[:, frame], args.risk_distance, args.max_pairs)
                if pairs:
                    for a, b, distance in pairs:
                        writer.writerow([frame, step, step * metadata["dt"], a, b, distance, int(truncated)])
                else:
                    writer.writerow([frame, step, step * metadata["dt"], "", "", "", 0])
    if args.save:
        if args.save.suffix.lower() == ".html":
            plt.rcParams["animation.embed_limit"] = 256
            args.save.write_text(anim.to_jshtml(fps=args.fps), encoding="utf-8")
        else:
            writer = animation.PillowWriter(fps=args.fps) if args.save.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=args.fps, bitrate=2200)
            anim.save(str(args.save), writer=writer, dpi=100)
        print(f"Animation: {args.save}")
    if args.snapshot:
        index = args.snapshot_frame if args.snapshot_frame >= 0 else positions.shape[1] - 1
        if index >= positions.shape[1]:
            parser.error("snapshot-frame is out of range")
        update(index)
        fig.savefig(args.snapshot, dpi=150, facecolor=fig.get_facecolor())
        print(f"Snapshot: {args.snapshot}")
    if not args.save and not args.snapshot:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
