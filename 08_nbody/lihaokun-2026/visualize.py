import argparse
import shutil
import struct
import time
from pathlib import Path

import numpy as np


OBJECT_STYLE = {
    "black_hole": ((255, 245, 120), 8),
    "neutron_star_white_dwarf": ((120, 220, 255), 4),
    "star": ((255, 170, 70), 3),
    "planet": ((80, 220, 130), 2),
    "asteroid": ((185, 195, 215), 1),
}


def load_trajectory(path: str):
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("trajectory file is too short")
        particles, records = struct.unpack("<ii", header)
    expected = 8 + records * particles * 3 * 4
    if particles <= 0 or records <= 0 or Path(path).stat().st_size != expected:
        raise ValueError(f"invalid trajectory dimensions: particles={particles}, records={records}")
    # File layout is particle-major: [particle, record, (x, y, z)].
    # Animation code uses frame-major: [record, particle, (x, y, z)].
    return np.memmap(path, dtype="<f4", mode="r", offset=8,
                     shape=(particles, records, 3)).transpose(1, 0, 2)


def axis_limits(values, z_scale=1.0):
    values = values.copy()
    if values.shape[-1] == 3:
        values[..., 2] *= z_scale
    flattened = values.reshape(-1, values.shape[-1])
    minimum = np.percentile(flattened, 1.0, axis=0)
    maximum = np.percentile(flattened, 99.0, axis=0)
    # Include the full initial state when it is not an outlier.
    minimum = np.minimum(minimum, values[0].min(axis=0))
    maximum = np.maximum(maximum, values[0].max(axis=0))
    center = (minimum + maximum) * 0.5
    half_range = max(float((maximum - minimum).max()) * 0.55, 1.0e-3)
    return [(c - half_range, c + half_range) for c in center]


def save_or_show(animation, output, dpi=120, fps=10, encoder_preset="veryfast"):
    if output:
        from matplotlib.animation import FFMpegWriter, PillowWriter
        suffix = Path(output).suffix.lower()
        writer = (PillowWriter(fps=fps) if suffix == ".gif" else
                  FFMpegWriter(fps=fps, codec="libx264", extra_args=[
                      "-preset", encoder_preset, "-crf", "20", "-pix_fmt", "yuv420p"]))
        animation.save(output, writer=writer, dpi=dpi)
    else:
        import matplotlib.pyplot as plt
        plt.show()


def animate_2d(trajectory, output=None, interval=100, trail=0, dpi=120,
               frames=None, trail_particles=48, width=1280, height=720,
               encoder_preset="veryfast"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    limits = axis_limits(trajectory[::max(1, len(trajectory) // 33), ::max(1, trajectory.shape[1] // 8192), :2])
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="x", ylabel="y", aspect="equal")
    points, = ax.plot([], [], "o", ms=4, alpha=0.85)
    ids = np.linspace(0, trajectory.shape[1] - 1,
                      min(trail_particles, trajectory.shape[1]) if trail else 0, dtype=int)
    trails = LineCollection([], linewidths=0.8, alpha=0.5)
    ax.add_collection(trails)
    title = ax.set_title("")

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        visible = trajectory[start:frame + 1]
        points.set_data(visible[-1, :, 0], visible[-1, :, 1])
        trails.set_segments(visible[:, ids, :2].transpose(1, 0, 2))
        title.set_text(f"N-body 2D | record {frame + 1}/{len(trajectory)}")
        return (points, title, trails)

    animation = FuncAnimation(fig, update, frames=frames if frames is not None else len(trajectory),
                              interval=interval, blit=False, cache_frame_data=False)
    save_or_show(animation, output, dpi, 1000 / interval, encoder_preset)
    plt.close(fig)


def animate_3d(trajectory, object_types=None, output=None, interval=100, trail=0,
               dpi=120, z_scale=1.0, frames=None, trail_particles=48,
               width=1280, height=720, azimuth=35, elevation=25,
               encoder_preset="veryfast"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    particles = trajectory.shape[1]
    if object_types is None:
        object_types = np.full(particles, "asteroid", dtype=object)
    draw_limit = 14000 if particles > 20000 else particles
    selected_groups = []
    for kind in OBJECT_STYLE:
        indices = np.flatnonzero(object_types == kind)
        if kind == "asteroid" and len(indices) > draw_limit:
            indices = indices[np.linspace(0, len(indices) - 1, draw_limit, dtype=int)]
        selected_groups.append(indices)
    selected = np.unique(np.concatenate(selected_groups))
    limit_frames = np.linspace(0, len(trajectory) - 1, min(len(trajectory), 33), dtype=int)
    limit_particles = np.linspace(0, len(selected) - 1,
                                  min(len(selected), 20000), dtype=int)
    limits = axis_limits(trajectory[np.ix_(limit_frames, selected[limit_particles])], z_scale)
    display = np.array(trajectory[0, selected], dtype=np.float32, copy=True)
    display[:, 2] *= z_scale

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#10131a")
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel="x", ylabel="y",
           zlabel=f"z (scale {z_scale:g})")
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_facecolor("#10131a")
    ax.tick_params(colors="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
        axis.set_pane_color((0.07, 0.09, 0.13, 1.0))
    ax.grid(True, alpha=0.22)
    title = ax.set_title("", color="white", pad=18, fontsize=13)

    selected_types = object_types[selected]
    scatters = {}
    locals_by_kind = {kind: np.flatnonzero(selected_types == kind) for kind in OBJECT_STYLE}
    for kind, (color, size) in OBJECT_STYLE.items():
        local = locals_by_kind[kind]
        if len(local):
            rgb = tuple(component / 255.0 for component in color)
            scatters[kind] = ax.scatter(
                display[local, 0], display[local, 1], display[local, 2],
                s=max(2.0, size * size * 0.75), c=[rgb],
                alpha=0.85 if kind != "asteroid" else 0.38,
                depthshade=False, edgecolors="none")

    # Batch all trail segments in one 3D collection. This shows substantially
    # more particle trajectories without creating one expensive Line3D object
    # per particle. Every body is still simulated and stored in the BIN file.
    trail_count = min(trail_particles, len(selected)) if trail else 0
    trail_indices = np.linspace(0, len(selected) - 1, trail_count, dtype=int)
    trail_collection = None
    if trail_indices.size:
        initial_points = display[trail_indices]
        initial_segments = np.stack((initial_points, initial_points), axis=1)
        trail_collection = Line3DCollection([], colors="#8ab4f8", linewidths=0.65,
                                            alpha=0.58)
        trail_collection.set_segments(initial_segments)
        ax.add_collection3d(trail_collection)
    legend_items = [
        Line2D([0], [0], marker="o", color="none", label=kind.replace("_", " "),
               markerfacecolor=np.array(color) / 255.0, markersize=max(4, size + 1))
        for kind, (color, size) in OBJECT_STYLE.items() if kind in scatters
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor="#202631", labelcolor="white", framealpha=0.85)

    def update(frame):
        start = max(0, frame - trail) if trail else frame
        for kind, points in scatters.items():
            local = locals_by_kind[kind]
            current = np.array(trajectory[frame, selected[local]], copy=True)
            current[:, 2] *= z_scale
            points._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
        if trail_collection is not None:
            history = np.array(trajectory[start:frame + 1, selected[trail_indices]], copy=True)
            history[:, :, 2] *= z_scale
            if len(history) > 1:
                trail_collection.set_segments(history.transpose(1, 0, 2))
            else:
                trail_collection.set_segments([])
        title.set_text(f"N-body 3D | frame {frame + 1}/{len(trajectory)} | "
                   f"bodies {len(selected):,}/{particles:,} | "
                   f"trails {len(trail_indices):,}")
        animated = (*scatters.values(), title)
        return animated + ((trail_collection,) if trail_collection is not None else ())

    animation = FuncAnimation(fig, update, frames=frames if frames is not None else len(trajectory),
                              interval=interval, blit=False, cache_frame_data=False)
    save_or_show(animation, output, dpi, 1000 / interval, encoder_preset)
    plt.close(fig)


def find_objects_file(trajectory_path, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"objects metadata not found: {path}")
        return path
    stem = Path(trajectory_path).stem
    candidates = (
        Path(trajectory_path).with_name(f"{stem}_objects.txt"),
        Path("data") / f"{stem}_objects.txt",
        Path(__file__).parent / "data" / f"{stem}_objects.txt",
    )
    return next((path for path in candidates if path.exists()), None)


def load_object_types(path, particles):
    names = np.full(particles, "asteroid", dtype=object)
    if path is None:
        return names
    rows = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    if len(rows) != particles:
        raise ValueError(f"metadata has {len(rows)} objects, trajectory has {particles}")
    seen = set()
    for row in rows:
        index, object_type, _mass = row.split()
        if object_type not in OBJECT_STYLE:
            raise ValueError(f"unknown object type: {object_type}")
        index = int(index)
        if index < 0 or index >= particles or index in seen:
            raise ValueError(f"invalid or duplicate metadata index: {index}")
        seen.add(index)
        names[index] = object_type
    return names


def main():
    from raster_animation import animate_raster, frame_indices

    parser = argparse.ArgumentParser(description="N-body: Matplotlib FuncAnimation with CUDA or CPU rendering")
    parser.add_argument("trajectory", nargs="?", default="trajectory.bin")
    parser.add_argument("--output", help="MP4 (recommended) or GIF; omit for interactive playback")
    parser.add_argument("--time-log")
    parser.add_argument("--objects", help="optional particle type metadata")
    parser.add_argument("--backend", choices=("auto", "gpu", "numpy", "matplotlib"), default="auto",
                        help="auto: CUDA if available, otherwise NumPy; matplotlib: native scatter/Axes3D")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-encoder", choices=("auto", "cpu", "nvenc"), default="auto",
                        help="auto/cpu use libx264 (A800 has no NVENC)")
    parser.add_argument("--encoder-preset", choices=("ultrafast", "superfast", "veryfast", "fast", "medium"),
                        default="veryfast")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--interval", type=float, help="milliseconds per frame; overrides --fps")
    parser.add_argument("--trail", type=int, default=80, help="history length in original records, not output frames")
    parser.add_argument("--trail-particles", type=int, default=48, help="maximum fixed particle IDs with path overlays")
    parser.add_argument("--dimension", choices=("2d", "3d"), default="3d")
    parser.add_argument("--layout", choices=("detail", "overview"), default="detail",
                        help="GPU/NumPy: overview + true local particle path and displacement curves")
    parser.add_argument("--focus-particle", type=int, help="GPU/NumPy: exact particle ID for local path (auto: largest sampled excursion)")
    parser.add_argument("--z-scale", type=float, default=1.0, help="display-only z scale, default preserves geometry")
    parser.add_argument("--azimuth", type=float, default=35)
    parser.add_argument("--elevation", type=float, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--raster-width", type=int, default=960)
    parser.add_argument("--raster-height", type=int, default=540)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--exposure", type=float, default=1.5, help="fixed particle density tone-map scale")
    parser.add_argument("--batch-frames", type=int, default=32, help="bounded contiguous trajectory upload batch")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="preview cap, evenly spaced across the full simulation")
    parser.add_argument("--view-radius", type=float, help="half-width of shorter camera axis in projected units")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), help="fixed camera center in input coordinates")
    parser.add_argument("--record-dt", type=float, help="simulation dt * record_interval, for time labels only")
    parser.add_argument("--time-unit", default="simulation units")
    args = parser.parse_args()
    for name in ("fps", "z_scale", "width", "height", "raster_width", "raster_height", "dpi",
                 "exposure", "batch_frames", "frame_stride"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("interval", "view_radius", "record_dt"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.trail < 0 or args.trail_particles < 0:
        parser.error("--trail and --trail-particles cannot be negative")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if not np.isfinite([args.azimuth, args.elevation]).all() or (args.center is not None and not np.isfinite(args.center).all()):
        parser.error("camera angles and center must be finite")
    if args.interval is not None:
        args.fps = 1000 / args.interval
    suffix = Path(args.output).suffix.lower() if args.output else ""
    if args.output:
        if suffix not in (".mp4", ".gif"):
            parser.error("--output must end with .mp4 or .gif")
        if suffix == ".mp4" and (args.width % 2 or args.height % 2):
            parser.error("H.264 requires even --width and --height")
        if suffix == ".mp4" and shutil.which("ffmpeg") is None:
            parser.error("MP4 export requires ffmpeg with libx264 in PATH")
        import matplotlib
        matplotlib.use("Agg")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    trajectory = load_trajectory(args.trajectory)
    if args.focus_particle is not None and not 0 <= args.focus_particle < trajectory.shape[1]:
        parser.error("--focus-particle is outside this trajectory's particle IDs")
    objects_path = find_objects_file(args.trajectory, args.objects)
    object_types = load_object_types(objects_path, trajectory.shape[1])
    frames = frame_indices(len(trajectory), args.frame_stride, args.max_frames)
    load_seconds = time.perf_counter() - total_start
    backend = args.backend
    if backend in ("auto", "gpu"):
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                device = torch.device(args.device)
                if device.type != "cuda":
                    parser.error("--device must select a CUDA device for GPU rendering")
                torch.cuda.get_device_properties(device)
        except ImportError:
            available = False
        if backend == "gpu" and not available:
            parser.error("--backend gpu requires CUDA-enabled PyTorch and a visible NVIDIA GPU")
        backend = "gpu" if available else "numpy"
        if backend == "numpy":
            print("CUDA unavailable; auto selected NumPy + FuncAnimation", flush=True)
    if args.video_encoder == "nvenc":
        if backend == "matplotlib":
            parser.error("native matplotlib mode uses libx264; choose --video-encoder cpu")
        if backend == "gpu" and any(k in torch.cuda.get_device_name(args.device).upper() for k in ("A100", "A800")):
            parser.error("A800/A100 have no NVENC; use --video-encoder cpu")
    if suffix == ".gif":
        print("GIF uses Pillow and buffers frames in host RAM; use MP4 for large/full-length runs.", flush=True)
    print(f"Records: {len(trajectory)}; output frames: {len(frames)}; playback: {len(frames) / args.fps:.2f}s", flush=True)
    if backend == "matplotlib":
        if args.focus_particle is not None:
            parser.error("--focus-particle requires the gpu or numpy backend")
        if args.view_radius is not None or args.center is not None:
            parser.error("--view-radius/--center require the gpu or numpy projection backend")
        if args.record_dt is not None:
            parser.error("--record-dt time labels require the gpu or numpy backend")
        if args.dimension == "3d":
            animate_3d(trajectory, object_types, args.output, 1000 / args.fps, args.trail,
                       args.dpi, args.z_scale, frames, args.trail_particles, args.width, args.height,
                       args.azimuth, args.elevation, args.encoder_preset)
        else:
            animate_2d(trajectory, args.output, 1000 / args.fps, args.trail, args.dpi,
                       frames, args.trail_particles, args.width, args.height, args.encoder_preset)
        stats = {"func_animation": True, "gpu_name": "none",
                 "video_encoder": "pillow" if suffix == ".gif" else "libx264"}
    else:
        stats = animate_raster(trajectory, object_types, OBJECT_STYLE, args, frames, backend)
    elapsed = time.perf_counter() - total_start
    if args.output:
        log_path = Path(args.time_log) if args.time_log else Path(args.output).with_suffix(".visualization.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "trajectory": args.trajectory, "output": args.output, "backend": backend,
            "dimension": args.dimension, "trajectory_frames": len(trajectory),
            "output_frames": len(frames), "frame_stride": args.frame_stride,
            "max_frames": args.max_frames, "particles": trajectory.shape[1],
            "objects_metadata": objects_path or "none", "fps": args.fps,
            "width": args.width, "height": args.height, "z_scale": args.z_scale,
            "trail_records": args.trail, "batch_frames": args.batch_frames,
            "encoder_preset": args.encoder_preset, "exposure": args.exposure,
            "output_size_bytes": Path(args.output).stat().st_size,
            "load_metadata_sec": load_seconds, **stats,
            "visualization_total_sec": elapsed, "export_frames_per_sec": len(frames) / elapsed,
        }
        log_path.write_text("\n".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                                      for key, value in report.items()) + "\n", encoding="utf-8")
        print(f"Visualization time: {elapsed:.3f}s; timing log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
