#!/usr/bin/env python3
"""visualize.py -- 读取二进制轨迹并生成 FuncAnimation 动画。

轨迹文件格式（需求要求 4，粒子主序）：
    int32 P            粒子数
    int32 R            记录点数
    float32 [P][R][3]  坐标，按粒子顺序，每粒子内按记录点顺序，每点 x,y,z

用法：
    python visualize.py traj.bin                      # 2D 动画，自动抽样
    python visualize.py traj.bin --dim 3              # 3D 动画
    python visualize.py traj.bin --trail 40           # 拖尾长度 40 个记录点
    python visualize.py traj.bin --save out.mp4       # 存文件（需 ffmpeg）
    python visualize.py traj.bin --save out.gif       # 存 GIF（用 pillow，无依赖）
    python visualize.py traj.bin --max-particles 3000 # 限制绘制粒子数
    python visualize.py traj.bin --frames 40           # 只渲染前 40 帧（快速预览）

拖尾的代价正比于“粒子数”（而不是拖尾长度）：所有拖尾装在一条
Line3DCollection 里，每帧只更新一个 artist，而不再是一个粒子一个 artist。
存文件时每 10 帧打印一次进度，不再出现“看起来卡死、其实在算”的情况。

若请求的后缀是视频但环境里没有 ffmpeg，会打印警告并自动改存同名 .gif，
而不是抛出一个看不出原因的 “unknown file extension: .mp4”。
"""

import argparse
import os
import sys

import numpy as np


def load_trajectory(path):
    """读取粒子主序二进制轨迹，返回 (R, P, 3) 的数组。

    文件是粒子主序 [P][R][3]，但动画按时间推进，每帧要取"所有粒子在
    某一记录点的位置"，所以转成记录主序 [R][P][3] 更自然。

    这个 transpose 是整个流程里最容易写反的一步：写反了不报错，
    但动画会完全错乱（每帧混合了不同时刻的不同粒子）。
    C++ 侧由 scripts/validate.sh 段 1 守着写出的布局（字节级交叉判定），这里守读入。
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"{path}: header is truncated")
        p, r = int(header[0]), int(header[1])
        if p <= 0 or r <= 0:
            raise ValueError(f"{path}: bad header P={p} R={r}")

        expected = p * r * 3
        data = np.fromfile(f, dtype=np.float32, count=expected)
        if data.size != expected:
            raise ValueError(
                f"{path}: expected {expected} floats, got {data.size} "
                f"(P={p}, R={r})"
            )

    # [P][R][3] -> [R][P][3]
    return data.reshape(p, r, 3).transpose(1, 0, 2)


def load_csv_trajectory(path):
    """读取 CSV 轨迹（particle_id,step,x,y,z），返回 (R, P, 3)。

    仅作对照。CSV 在 N=65536 时是几百 MB 的文本，解析要几十秒，
    而同样内容的二进制不到 1 秒 —— 这就是报告里"CSV 效率较低"的依据。
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    pid = raw[:, 0].astype(np.int64)
    step = raw[:, 1].astype(np.int64)
    uniq_steps = np.unique(step)
    p = int(pid.max()) + 1
    r = uniq_steps.size
    step_index = {s: i for i, s in enumerate(uniq_steps)}
    out = np.zeros((r, p, 3), dtype=np.float32)
    for row in raw:
        out[step_index[int(row[1])], int(row[0])] = row[2:5]
    return out


def choose_subset(num_particles, limit, seed=0):
    """抽样出要绘制的粒子索引。

    matplotlib 在每帧重绘几万个点时会慢到不可用（N=65536 时约 1 fps），
    所以超过 limit 就随机抽样。抽样而非取前 limit 个：初值文件里粒子
    可能按半径或质量排序（例如 disk 预设第 0 个是中心核球），
    取前 N 个会得到有偏的子集，动画看起来像是只有星系中心。
    """
    if num_particles <= limit:
        return np.arange(num_particles)
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_particles, size=limit, replace=False)
    idx.sort()
    return idx


def robust_limits(positions, dim, percentile=99.0, pad=0.1):
    """按分位数确定坐标轴范围。

    不用 min/max：引力模拟里常有个别粒子被弹射到极远处，
    用真实极值会把整个星团压成中心一个点。
    """
    flat = positions.reshape(-1, 3)[:, :dim]
    finite = flat[np.isfinite(flat).all(axis=1)]
    if finite.size == 0:
        return [(-1.0, 1.0)] * dim
    lo = np.percentile(finite, 100.0 - percentile, axis=0)
    hi = np.percentile(finite, percentile, axis=0)
    # 各轴取同一个半宽，避免不等比缩放让圆轨道看起来像椭圆
    center = (lo + hi) / 2.0
    half = float(np.max(hi - lo)) / 2.0
    if half <= 0.0 or not np.isfinite(half):
        half = 1.0
    half *= 1.0 + pad
    return [(float(center[k] - half), float(center[k] + half)) for k in range(dim)]


VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".avi", ".mkv")
GIF_EXTS = (".gif",)


def _progress(frame, total):
    """渲染进度。没有这个输出时，慢渲染看起来和卡死一模一样。"""
    if (frame + 1) % 10 == 0 or frame + 1 == total:
        print(f"\r  rendering frame {frame + 1}/{total}", end="", flush=True)
        if frame + 1 == total:
            print()


def write_animation(anim, path, fps, dpi):
    """存动画，返回实际写出的文件名。

    容器里常常没有 ffmpeg，而 matplotlib 在找不到 ffmpeg 时会**静默降级到
    Pillow**，再由 Pillow 抛出 "unknown file extension: .mp4" —— 报错信息
    与真正的原因（缺 ffmpeg）看不出关系。这里把它变成一次显式判断：
    请求视频后缀但没有 ffmpeg，就改存同名 .gif 并说清楚。
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    ext = os.path.splitext(path)[1].lower()
    if ext in GIF_EXTS:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                  progress_callback=_progress)
        return path

    if ext in VIDEO_EXTS and not FFMpegWriter.isAvailable():
        fallback = os.path.splitext(path)[0] + ".gif"
        print(f"warning: ffmpeg not found -> cannot write {ext}\n"
              f"         falling back to {fallback}\n"
              f"         (install ffmpeg for {ext}, e.g. apt-get install -y "
              f"ffmpeg)")
        anim.save(fallback, writer=PillowWriter(fps=fps), dpi=dpi,
                  progress_callback=_progress)
        return fallback

    anim.save(path, fps=fps, dpi=dpi, progress_callback=_progress)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="trajectory file (.bin or .csv)")
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3),
                    help="2D or 3D animation (default 2)")
    ap.add_argument("--trail", type=int, default=0,
                    help="draw a fading trail of this many past records "
                         "(0 = points only; cost grows with the number of "
                         "particles drawn, so pair it with --max-particles)")
    ap.add_argument("--frames", type=int, default=0,
                    help="animate only the first N records (0 = all); "
                         "useful for a quick preview before the full render")
    ap.add_argument("--max-particles", type=int, default=2000,
                    help="cap on particles drawn (random subset; default 2000)")
    ap.add_argument("--interval", type=int, default=30,
                    help="delay between frames in ms (default 30)")
    ap.add_argument("--size", type=float, default=2.0, help="marker size")
    ap.add_argument("--save", default=None,
                    help="write animation to .mp4 (ffmpeg) or .gif (pillow) "
                         "instead of showing a window")
    ap.add_argument("--fps", type=int, default=30, help="fps when saving")
    ap.add_argument("--dpi", type=int, default=110, help="dpi when saving")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"error: no such file: {args.path}")

    if args.path.lower().endswith(".csv"):
        positions = load_csv_trajectory(args.path)
    else:
        positions = load_trajectory(args.path)

    num_records, num_particles, _ = positions.shape
    print(f"loaded {args.path}: P={num_particles} R={num_records}")

    if args.frames and args.frames < num_records:
        positions = positions[:args.frames]
        num_records = args.frames
        print(f"limiting the animation to the first {num_records} records (--frames)")

    subset = choose_subset(num_particles, args.max_particles)
    if subset.size < num_particles:
        print(f"drawing a random subset of {subset.size}/{num_particles} particles "
              f"(matplotlib cannot animate all of them at a usable frame rate)")
    positions = positions[:, subset, :]

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    limits = robust_limits(positions, args.dim)

    if args.dim == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # 3D 散点没有 set_offsets，只能用 _offsets3d，是私有属性但这是
        # matplotlib 里公认的做法（官方 example 也这么用）。
        scat = ax.scatter(positions[0, :, 0], positions[0, :, 1],
                          positions[0, :, 2], s=args.size, c="#4da6ff",
                          depthshade=False)
        trails = None
        if args.trail > 0:
            # 所有拖尾装在一条 Line3DCollection 里。早先是“一个粒子一个
            # ax.plot()”：4096 粒子就是 4096 个 artist，每帧逐条 set_data +
            # set_3d_properties，200 帧共 80 万次 Python 层更新，看起来像卡死。
            trails = Line3DCollection([], linewidths=0.4, colors="#4da6ff",
                                      alpha=0.45)
            ax.add_collection3d(trails, autolim=False)
    else:
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=args.size,
                          c="#4da6ff")
        trails = None
        if args.trail > 0:
            trails = LineCollection([], linewidths=0.5, colors="#4da6ff",
                                    alpha=0.5)
            ax.add_collection(trails)

    # 坐标范围最后设：add_collection 会按（空的）数据做一次 autoscale，
    # 先设会被它覆盖掉。
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    if args.dim == 3:
        ax.set_zlim(*limits[2])

    ax.set_title(f"N-body: {num_particles} particles, record 0/{num_records - 1}")
    fig.tight_layout()

    def update(frame):
        artists = [scat]
        if args.dim == 3:
            scat._offsets3d = (positions[frame, :, 0], positions[frame, :, 1],
                               positions[frame, :, 2])
        else:
            # set_offsets 而非重新 scatter：重画会不断累积 artist，
            # 几十帧后就慢得不能看。
            scat.set_offsets(positions[frame, :, :2])

        if trails is not None:
            start = max(0, frame - args.trail)
            segs = np.moveaxis(positions[start:frame + 1], 1, 0)
            if args.dim == 2:
                segs = segs[..., :2]
            trails.set_segments(list(segs))
            artists.append(trails)

        ax.set_title(f"N-body: {num_particles} particles, "
                     f"record {frame}/{num_records - 1}")
        return artists

    anim = FuncAnimation(fig, update, frames=num_records,
                         interval=args.interval, blit=False, repeat=True)

    if args.save:
        print(f"wrote {write_animation(anim, args.save, args.fps, args.dpi)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
