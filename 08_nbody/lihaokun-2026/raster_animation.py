"""CUDA/NumPy rasterization inside a real Matplotlib FuncAnimation.

The camera is an orthographic projection, not an interactive Axes3D. All bodies
contribute to the image; only the explicitly labelled trajectory overlay is
sampled. Matplotlib composition and libx264 encoding still run on the CPU.
"""
import time

import numpy as np


BACKGROUND = np.array([10, 16, 27], dtype=np.float32) / 255


def frame_indices(records, stride=1, max_frames=None):
    """Temporal subsampling always includes both endpoints (also for previews)."""
    frames = np.arange(0, records, stride, dtype=int)
    if frames[-1] != records - 1:
        frames = np.append(frames, records - 1)
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames, dtype=int)]
    return frames


def projection_matrix(dimension="3d", azimuth=35, elevation=25, z_scale=1):
    if dimension == "2d":
        return np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    a, e = np.deg2rad([azimuth, elevation])
    return np.array([[np.cos(a), -np.sin(e) * np.sin(a)],
                     [-np.sin(a), -np.sin(e) * np.cos(a)],
                     [0, np.cos(e) * z_scale]], dtype=np.float32)


def camera_limits(trajectory, matrix, width, height, radius=None, center=None):
    # Bounded sample, fixed camera, equal physical scale in both image axes.
    fs = np.linspace(0, len(trajectory) - 1, min(33, len(trajectory)), dtype=int)
    ps = np.linspace(0, trajectory.shape[1] - 1,
                     min(8192, trajectory.shape[1]), dtype=int)
    sample = np.asarray(trajectory[np.ix_(fs, ps)]) @ matrix
    finite = sample[np.isfinite(sample).all(axis=-1)]
    if not len(finite):
        raise ValueError("camera sample contains no finite coordinates")
    low, high = np.percentile(finite, [0.2, 99.8], axis=0)
    middle = (low + high) / 2 if center is None else np.asarray(center) @ matrix
    half = max(float(np.max(np.abs(np.stack((low, high)) - middle))), 1e-6) * 1.08
    if radius is not None:
        half = radius
    aspect = width / height
    half_xy = np.array([half * max(1, aspect), half * max(1, 1 / aspect)])
    return np.asarray(middle - half_xy, dtype=np.float32), np.asarray(middle + half_xy, dtype=np.float32)


class Rasterizer:
    """One bilinear scatter per frame, without per-pixel Python/CUDA launches.

    Four accumulators hold RGB sums and particle density. Log tone mapping uses
    a fixed exposure, preserving changes in density rather than normalizing each
    frame independently. Out-of-view/non-finite positions go to a sentinel bin.
    """
    def __init__(self, matrix, low, high, colors, width, height, exposure=1.5,
                 device=None):
        self.width, self.height = width, height
        self.exposure = exposure
        self.device = device
        self.torch = None
        self.matrix, self.low, self.high = matrix, low, high
        self.scale = np.array([width, height], dtype=np.float32) / (high - low)
        self.colors = np.column_stack((colors, np.ones(len(colors)))).astype(np.float32)
        self.offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
        if device is not None:
            import torch
            self.torch = torch
            for name in ("matrix", "low", "high", "scale", "colors", "offsets"):
                setattr(self, name, torch.as_tensor(getattr(self, name), device=device))
            self.background = torch.as_tensor(BACKGROUND, device=device)
            self.canvas = torch.zeros((width * height + 1, 4), device=device)
        else:
            self.canvas = np.zeros((width * height + 1, 4), dtype=np.float32)

    def render(self, points):
        if self.torch is not None:
            return self._torch_render(points)
        projected = points @ self.matrix
        valid = np.isfinite(projected).all(axis=1)
        valid &= ((projected >= self.low) & (projected < self.high)).all(axis=1)
        xy = np.where(valid[:, None], (projected - self.low) * self.scale - 0.5, 0)
        base = np.floor(xy).astype(np.int64)
        frac = xy - base
        cells = base[:, None, :] + self.offsets
        weights = np.prod(np.where(self.offsets[None] == 1,
                                   frac[:, None], 1 - frac[:, None]), axis=-1)
        inside = valid[:, None] & (cells[:, :, 0] >= 0) & (cells[:, :, 0] < self.width)
        inside &= (cells[:, :, 1] >= 0) & (cells[:, :, 1] < self.height)
        indices = np.where(inside, cells[:, :, 1] * self.width + cells[:, :, 0],
                           self.width * self.height)
        self.canvas.fill(0)
        values = (weights[:, :, None] * self.colors[:, None, :]).reshape(-1, 4)
        np.add.at(self.canvas, indices.ravel(), values)
        sums = self.canvas[:-1]
        density = sums[:, 3:4]
        hue = sums[:, :3] / np.maximum(density, 1e-8)
        alpha = np.clip(np.log1p(density) / np.log1p(self.exposure), 0, 1)
        rgb = BACKGROUND + alpha * (hue - BACKGROUND)
        return np.rint(rgb.reshape(self.height, self.width, 3) * 255).astype(np.uint8)

    def _torch_render(self, points):
        t = self.torch
        points = t.as_tensor(points, dtype=t.float32, device=self.device)
        projected = points @ self.matrix
        valid = t.isfinite(projected).all(dim=1)
        valid &= ((projected >= self.low) & (projected < self.high)).all(dim=1)
        xy = t.where(valid[:, None], (projected - self.low) * self.scale - 0.5, 0)
        base = t.floor(xy).long()
        frac = xy - base
        cells = base[:, None, :] + self.offsets
        weights = t.where(self.offsets[None] == 1,
                          frac[:, None], 1 - frac[:, None]).prod(dim=-1)
        inside = valid[:, None] & (cells[:, :, 0] >= 0) & (cells[:, :, 0] < self.width)
        inside &= (cells[:, :, 1] >= 0) & (cells[:, :, 1] < self.height)
        indices = t.where(inside, cells[:, :, 1] * self.width + cells[:, :, 0],
                          self.width * self.height)
        self.canvas.zero_()
        values = (weights[:, :, None] * self.colors[:, None, :]).reshape(-1, 4)
        self.canvas.index_add_(0, indices.reshape(-1), values)
        sums = self.canvas[:-1]
        density = sums[:, 3:4]
        hue = sums[:, :3] / density.clamp_min(1e-8)
        alpha = (t.log1p(density) / np.log1p(self.exposure)).clamp(0, 1)
        rgb = self.background + alpha * (hue - self.background)
        return (rgb.reshape(self.height, self.width, 3) * 255).round().byte().cpu().numpy()


class FrameBuffer:
    """Bounded contiguous record batches instead of duplicating a whole BIN."""
    def __init__(self, trajectory, batch_frames, device=None):
        self.trajectory, self.batch_frames, self.device = trajectory, batch_frames, device
        self.start, self.stop = -1, -1
        self.buffer = None
        self.load_seconds = 0.0

    def get(self, frame):
        if not self.start <= frame < self.stop:
            start = time.perf_counter()
            self.start = frame
            self.stop = min(frame + self.batch_frames, len(self.trajectory))
            # copy=True is essential: memmaps are read-only and may already be contiguous.
            host = np.array(self.trajectory[self.start:self.stop], dtype=np.float32,
                            order="C", copy=True)
            self.buffer = None
            if self.device is not None:
                import torch
                self.buffer = torch.from_numpy(host).to(self.device)
                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize(self.device)
            else:
                self.buffer = host
            self.load_seconds += time.perf_counter() - start
        return self.buffer[frame - self.start]


def select_tracers(trajectory, matrix, low, high, count):
    """Fixed IDs, spread over initial radius; never switch identities per frame."""
    first = np.asarray(trajectory[0]) @ matrix
    valid = np.isfinite(first).all(axis=1) & ((first >= low) & (first <= high)).all(axis=1)
    ids = np.flatnonzero(valid)
    if not len(ids) or count == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(np.linalg.norm(first[ids] - (low + high) / 2, axis=1))
    return ids[order[np.linspace(0, len(ids) - 1, min(count, len(ids)), dtype=int)]]


def focus_path(trajectory, particle=None):
    """Select real motion and return world coordinates relative to the start.

    An explicit ID is never sampled. Auto selects the largest sampled excursion
    (up to 8192 IDs/33 records), not a claim about the globally fastest body.
    """
    if particle is None:
        ids = np.linspace(0, trajectory.shape[1] - 1, min(8192, trajectory.shape[1]), dtype=int)
        frames = np.linspace(0, len(trajectory) - 1, min(33, len(trajectory)), dtype=int)
        sample = np.asarray(trajectory[np.ix_(frames, ids)], dtype=np.float64)
        distances = np.linalg.norm(sample - sample[0], axis=-1)
        score = np.where(np.isfinite(distances), distances, -np.inf).max(axis=0)
        particle = int(ids[np.argmax(score)])
    if not 0 <= particle < trajectory.shape[1]:
        raise ValueError(f"focus particle ID must be in [0, {trajectory.shape[1] - 1}]")
    path = np.array(trajectory[:, particle], dtype=np.float64, copy=True)
    if not np.isfinite(path).all():
        raise ValueError("focus path has non-finite coordinates; validate the trajectory")
    return particle, path - path[0]


def animate_raster(trajectory, object_types, styles, args, frames, backend):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    setup_start = time.perf_counter()
    device = args.device if backend == "gpu" else None
    gpu_name = "none"
    if device is not None:
        import torch
        gpu_name = torch.cuda.get_device_name(device)
    # Rendering resolution is independent of the number of particles.
    rw, rh = args.raster_width, args.raster_height
    matrix = projection_matrix(args.dimension, args.azimuth, args.elevation, args.z_scale)
    low, high = camera_limits(trajectory, matrix, rw, rh, args.view_radius, args.center)
    colors = np.array([styles[k][0] for k in object_types], dtype=np.float32) / 255
    renderer = Rasterizer(matrix, low, high, colors, rw, rh, args.exposure, device)
    buffer = FrameBuffer(trajectory, args.batch_frames, device)
    trace_ids = select_tracers(trajectory, matrix, low, high,
                              args.trail_particles if args.trail else 0)
    # Small CPU overlay only. Every body still goes through the GPU rasterizer.
    highlighted = np.flatnonzero(object_types == "black_hole")[:32]
    overlay_ids = np.unique(np.concatenate((trace_ids, highlighted)))
    trace_local = np.searchsorted(overlay_ids, trace_ids)
    special_local = np.searchsorted(overlay_ids, highlighted)
    history = np.asarray(trajectory[:, overlay_ids]) @ matrix
    focus_id, relative_path = focus_path(trajectory, args.focus_particle)
    relative_projection = relative_path @ matrix
    sample_ids = np.linspace(0, trajectory.shape[1] - 1,
                             min(8192, trajectory.shape[1]), dtype=int)
    sample_frames = np.linspace(0, len(trajectory) - 1, min(33, len(trajectory)), dtype=int)
    motion = np.asarray(trajectory[np.ix_(sample_frames, sample_ids)]) @ matrix
    motion_pixels = np.linalg.norm((motion - motion[0]) * np.array([rw, rh]) / (high - low), axis=-1)
    finite_motion = np.max(motion_pixels, axis=0)
    finite_motion = finite_motion[np.isfinite(finite_motion)]
    median_motion = float(np.median(finite_motion)) if len(finite_motion) else float("nan")
    sampled_visible_fraction = float(np.mean(np.isfinite(motion).all(axis=-1) &
                                             ((motion >= low) & (motion < high)).all(axis=-1)))
    print(f"Backend: {backend}; GPU: {gpu_name}; FuncAnimation: yes", flush=True)
    print(f"Sampled median maximum motion: {median_motion:.3f} raster pixels", flush=True)
    if median_motion < 2:
        print("Motion is below 2 pixels for most sampled bodies. Use --view-radius with --center "
              "for a local view, or the orbit demo; playback FPS cannot create missing motion.", flush=True)

    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi),
                     dpi=args.dpi, facecolor="#0a101b")
    if args.layout == "detail":
        ax = fig.add_axes([0.065, 0.18, 0.53, 0.65])
        detail = fig.add_axes([0.69, 0.55, 0.27, 0.29])
        series = fig.add_axes([0.69, 0.20, 0.27, 0.20])
    else:
        ax = fig.add_axes([0.08, 0.18, 0.88, 0.65])
    ax.set_facecolor("#0a101b")
    ax.tick_params(colors="#aebcd0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#35445b")
    label_x, label_y = ("x", "y") if args.dimension == "2d" else ("projected u", "projected v")
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
           xlabel=f"{label_x} (input length units)", ylabel=f"{label_y} (input length units)")
    ax.xaxis.label.set_color("#aebcd0")
    ax.yaxis.label.set_color("#aebcd0")
    image = ax.imshow(np.zeros((rh, rw, 3), dtype=np.uint8), origin="lower",
                      extent=(low[0], high[0], low[1], high[1]), interpolation="nearest")
    trace_colors = plt.get_cmap("cool")(np.linspace(0.15, 0.85, max(1, len(trace_ids))))
    lines = LineCollection([], colors=trace_colors, linewidths=1.0, alpha=0.8)
    ax.add_collection(lines)
    markers = ax.scatter([], [], s=10, color="#f7e8ff", edgecolors="none")
    specials = ax.scatter([], [], s=65, color="#fff578", edgecolors="#ffffff", linewidths=0.5)
    focus_world = np.asarray(trajectory[:, focus_id], dtype=np.float64) @ matrix
    focus_marker = ax.scatter([], [], s=100, facecolors="none", edgecolors="#ffe089", linewidths=1.5)
    title = ax.set_title("", color="white", fontsize=10, pad=10)
    mode = "XY" if args.dimension == "2d" else "3D orthographic"
    fig.text(0.065, 0.94, f"N-BODY / {mode} / {trajectory.shape[1]:,} bodies", color="white", fontsize=15, weight="bold")
    fig.text(0.065, 0.89, f"Fixed camera | {len(trace_ids)} tracked paths | z scale {args.z_scale:g} | "
             f"sampled median excursion {median_motion:.2f} px", color="#a5b9d1", fontsize=9)
    legend = [Line2D([], [], marker="o", linestyle="none", markersize=4,
                     color=np.array(color) / 255, label=kind.replace("_", " "))
              for kind, (color, _) in styles.items() if np.any(object_types == kind)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=len(legend), frameon=False, labelcolor="#aebcd0", fontsize=8)
    artists = (image, lines, markers, specials, focus_marker, title)
    if args.layout == "detail":
        for panel in (detail, series):
            panel.set_facecolor("#0d1828")
            panel.tick_params(colors="#aebcd0", labelsize=7)
            panel.grid(alpha=0.12)
            for spine in panel.spines.values():
                spine.set_color("#35445b")
            panel.xaxis.label.set_color("#aebcd0")
            panel.yaxis.label.set_color("#aebcd0")
        bounds = np.array([relative_projection.min(axis=0), relative_projection.max(axis=0)])
        middle = bounds.mean(axis=0)
        half = max(float(np.ptp(bounds, axis=0).max()) * .65, 1e-9)
        detail.set(xlim=(middle[0] - half, middle[0] + half), ylim=(middle[1] - half, middle[1] + half),
                   aspect="equal", xlabel="u - u(initial)", ylabel="v - v(initial)")
        detail.plot(relative_projection[:, 0], relative_projection[:, 1], color="#56677e", lw=1.1)
        detail.plot([0], [0], "o", color="#8faecc", ms=4)
        focus_line, = detail.plot([], [], color="#6fe1c7", lw=1.8)
        focus_dot, = detail.plot([], [], "o", color="#ffe089", ms=7)
        detail_title = detail.set_title(f"Particle #{focus_id} | real local path", color="white", fontsize=9)
        record_times = np.arange(len(trajectory)) * (args.record_dt or 1)
        for k, (color, label) in enumerate(zip(("#ef8993", "#6bd9a5", "#83b4ff"), ("dx", "dy", "dz"))):
            series.plot(record_times, relative_path[:, k], color=color, lw=1, label=label)
        series.set(xlabel=args.time_unit if args.record_dt else "record", ylabel="displacement (input units)")
        series.legend(loc="upper left", ncol=3, frameon=False, labelcolor="#aebcd0", fontsize=7)
        cursor = series.axvline(0, color="#ffe089", lw=1)
        fig.text(.69, .46, "Local axes zoom into real displacement.\nNo artificial motion scaling.", color="#a5b9d1", fontsize=8)
        artists += (focus_line, focus_dot, detail_title, cursor)
    render_seconds = 0.0
    rendered = 0
    frame_times = []
    setup_seconds = time.perf_counter() - setup_start
    save_start = time.perf_counter()

    def init():
        return artists

    def update(frame):
        nonlocal render_seconds, rendered
        begin = time.perf_counter()
        points = buffer.get(int(frame))
        render_start = time.perf_counter()
        rgb = renderer.render(points)  # .cpu() synchronizes the GPU before timing ends.
        render_seconds += time.perf_counter() - render_start
        image.set_data(rgb)
        start = max(0, int(frame) - args.trail)
        lines.set_segments(history[start:int(frame) + 1, trace_local].transpose(1, 0, 2))
        markers.set_offsets(history[frame, trace_local])
        specials.set_offsets(history[frame, special_local])
        focus_marker.set_offsets(focus_world[frame:frame + 1])
        time_label = (f" | t={frame * args.record_dt:.6g} {args.time_unit}"
                      if args.record_dt is not None else "")
        title.set_text(f"record {frame + 1}/{len(trajectory)}{time_label}")
        if args.layout == "detail":
            focus_line.set_data(relative_projection[:frame + 1, 0], relative_projection[:frame + 1, 1])
            focus_dot.set_data([relative_projection[frame, 0]], [relative_projection[frame, 1]])
            detail_title.set_text(f"Particle #{focus_id} | distance {np.linalg.norm(relative_path[frame]):.5g}")
            cursor.set_xdata([record_times[frame], record_times[frame]])
        frame_times.append(time.perf_counter() - begin)
        rendered += 1
        if rendered == 1 or rendered % 25 == 0 or rendered == len(frames):
            elapsed = time.perf_counter() - save_start
            print(f"Rendered {rendered}/{len(frames)} | elapsed {elapsed:.1f}s", flush=True)
        return artists

    animation = FuncAnimation(fig, update, frames=frames, init_func=init,
                              interval=1000 / args.fps, blit=False,
                              cache_frame_data=False)
    encoder = "none"
    try:
        if args.output:
            if args.output.lower().endswith(".gif"):
                writer = PillowWriter(fps=args.fps)
                encoder = "pillow"
            else:
                # A800 has no NVENC. CPU is the portable and explicit default.
                encoder = "h264_nvenc" if args.video_encoder == "nvenc" else "libx264"
                extra = (["-preset", "p4", "-cq", "21"] if encoder == "h264_nvenc" else
                         ["-preset", args.encoder_preset, "-crf", "20"])
                writer = FFMpegWriter(fps=args.fps, codec=encoder,
                                      extra_args=extra + ["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            animation.save(args.output, writer=writer, dpi=args.dpi)
        else:
            plt.show()
    finally:
        plt.close(fig)
    save_seconds = time.perf_counter() - save_start
    return {
        "gpu_name": gpu_name, "video_encoder": encoder, "func_animation": True,
        "raster_width": rw, "raster_height": rh, "trail_particles": len(trace_ids),
        "sampled_median_max_motion_pixels": median_motion,
        "sampled_visible_fraction": sampled_visible_fraction,
        "layout": args.layout, "focus_particle_id": focus_id,
        "setup_sec": setup_seconds, "frame_load_and_upload_sec": buffer.load_seconds,
        "raster_and_readback_sec": render_seconds,
        "matplotlib_encode_other_sec": max(0, save_seconds - render_seconds - buffer.load_seconds),
        "animation_save_sec": save_seconds,
        "update_p50_ms": float(np.median(frame_times) * 1000) if frame_times else 0,
        "camera_low": low.tolist(), "camera_high": high.tolist(),
    }
