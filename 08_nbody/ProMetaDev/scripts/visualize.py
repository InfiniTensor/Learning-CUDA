#!/usr/bin/env python3
"""
CudaNBodyGravitySim_2026summer —— 轨迹可视化脚本

题面轨迹二进制格式 (推荐)
---------------------------
  header: int32 N, int32 R
  frames: R frames × N×3 float32 (粒子按 0..N-1 顺序, x y z 连续)

CSV 格式 (可选, 低效)
---------------------------
  particle_id, step, x, y, z

用法 (两种模式皆可):
  # 模式 A: 位置参数 (README 推荐写法)
  python3 scripts/visualize.py demo 2body
  python3 scripts/visualize.py demo 3body --out out.gif
  python3 scripts/visualize.py demo cluster --out outputs/cluster_demo.gif

  # 模式 B: --demo flag (旧写法, 完全兼容)
  python3 scripts/visualize.py --demo cluster --out outputs/cluster_demo.gif

  # 二进制 -> 2D GIF
  python3 scripts/visualize.py outputs/trajectory.bin --out outputs/two_body.gif --interval 30

  # 65k 粒子 3D 预览 (下采样到 8000 点 + blit 加速)
  python3 scripts/visualize.py outputs/65k.bin --view 3d --frames 11 \
      --max-points 8000 --blit --out outputs/65k.gif --fmt gif

  # 二进制 -> MP4 (需要系统 ffmpeg, 文件比 GIF 小 80%)
  python3 scripts/visualize.py outputs/65k.bin --fmt mp4 --out outputs/65k.mp4

  # CSV 输入 (visualize 根据 .csv 后缀自动识别; 也可以显式 --csv)
  python3 scripts/visualize.py outputs/t3.bin.csv --out outputs/t3_fromcsv.gif

  # 播放 (不保存)
  python3 scripts/visualize.py outputs/trajectory.bin --view 3d
"""
from __future__ import annotations
import argparse, struct, sys, os, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg" if (os.environ.get("DISPLAY") or sys.platform.startswith("win")) else "Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# 二进制轨迹读 (题面格式, 小端序)
# ----------------------------------------------------------------------
def load_binary_trajectory(path: str):
    """returns (N, R, frames) where frames.shape = (R, N, 3), dtype float64"""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise RuntimeError("file too short (no header)")
    N = struct.unpack_from("<i", data, 0)[0]
    R = struct.unpack_from("<i", data, 4)[0]
    need = 8 + R * N * 3 * 4
    if len(data) < need:
        avail = len(data) - 8
        R_clamped = avail // (N * 3 * 4)
        print(f"[WARN] truncated file: expect {need} bytes, got {len(data)}. "
              f"Clamp R from {R} to {R_clamped}.")
        R = R_clamped
    arr = np.frombuffer(data, dtype="<f4", count=R*N*3, offset=8).astype(np.float64)
    frames = arr.reshape(R, N, 3)
    lo = frames.min(axis=(0,1)); hi = frames.max(axis=(0,1))
    print(f"[load.bin] N={N}, R={R}, x=[{lo[0]:.4f},{hi[0]:.4f}]  "
          f"y=[{lo[1]:.4f},{hi[1]:.4f}]  z=[{lo[2]:.4f},{hi[2]:.4f}]")
    return N, R, frames

# ----------------------------------------------------------------------
# CSV 轨迹读 (N,R,frames)
# ----------------------------------------------------------------------
def load_csv_trajectory(path: str):
    print(f"[load.csv] reading {path} (this may take a while)...")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:  arr = arr[None, :]
    pids = arr[:, 0].astype(np.int64)
    steps = arr[:, 1].astype(np.int64)
    xyz = arr[:, 2:5].astype(np.float64)
    N = int(pids.max()) + 1
    step_set = sorted(set(steps.tolist()))
    # 按 step, pid 对映
    step_to_idx = {s:i for i,s in enumerate(step_set)}
    R = len(step_set)
    frames = np.empty((R, N, 3), dtype=np.float64)
    frames[:] = np.nan
    for k in range(arr.shape[0]):
        frames[step_to_idx[steps[k]], pids[k], :] = xyz[k]
    bad = np.isnan(frames).sum()
    if bad: print(f"[load.csv] WARN {bad} NaN entries (missing rows)")
    lo = np.nanmin(frames, axis=(0,1)); hi = np.nanmax(frames, axis=(0,1))
    print(f"[load.csv] N={N}, R={R},  x=[{lo[0]:.4f},{hi[0]:.4f}]  y=[{lo[1]:.4f},{hi[1]:.4f}]")
    return N, R, frames

# ----------------------------------------------------------------------
# 演示模式 —— 内置简易 Leapfrog (纯 NumPy)
# ----------------------------------------------------------------------
def leapfrog_demo(name: str, R=240, dt=0.02, G=1.0, eps=0.02):
    if name == "2body":
        M = np.array([1.0, 1.0], dtype=np.float64)
        X = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
        v0 = math.sqrt(G * M.sum() / 1.0**3) * 0.5
        V = np.array([[0.0, v0, 0.0], [0.0, -v0, 0.0]], dtype=np.float64)
    elif name == "3body":
        M = np.array([1.0, 1.0, 0.01])
        X = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float64)
        v_bin = math.sqrt(2.0) * 0.5
        v_out = math.sqrt(G * 2.01 / 10.0)
        V = np.array([[0.0, v_bin, 0.0], [0.0, -v_bin, 0.0], [-v_out, 0.0, 0.0]])
    elif name == "cluster":
        rng = np.random.default_rng(2026)
        N = 300
        M = rng.uniform(0.7, 1.3, N)
        pts = rng.normal(size=(N, 3))
        rs = np.linalg.norm(pts, axis=1, keepdims=True)
        rs = np.where(rs == 0, 1, rs)
        X = (pts / rs) * (rng.uniform(0, 1, size=(N, 1)) ** (1/3)) * 2.0
        V = rng.normal(size=(N, 3))
        for _ in range(3):
            sc = 0.2 / np.linalg.norm(V, axis=1, keepdims=True).mean()
            V *= sc
        X -= (M[:, None] * X).sum(axis=0, keepdims=True) / M.sum()
        V -= (M[:, None] * V).sum(axis=0, keepdims=True) / M.sum()
    else:
        raise ValueError(f"unknown demo {name}")

    N = len(M)
    frames = np.empty((R, N, 3), dtype=np.float64)
    frames[0] = X.copy()

    def accel(X):
        dX = X[None, :, :] - X[:, None, :]                # N×N×3
        r2 = (dX * dX).sum(axis=2) + eps * eps            # N×N
        inv_r3 = r2 ** (-1.5)
        np.fill_diagonal(inv_r3, 0.0)
        A = G * (inv_r3[:, :, None] * dX * M[None, :, None]).sum(axis=1)
        return A

    A = accel(X)
    # Kick v_half = v0 + a0*dt/2
    V = V + A * (dt * 0.5)
    for k in range(1, R):
        X += V * dt
        A_new = accel(X)
        # 完整速度仅用于能量, 这里每步按 KDK 继续: V 保持半速
        # 若需要记录全速, 可再加 0.5*A_new*dt, 然后下一步 V_full 再转半速.
        # 这里简化: 半速 + 下一半步.
        V = V + A_new * dt
        frames[k] = X.copy()
    print(f"[demo {name}] N={N}, R={R}, dt={dt}, G={G}, eps={eps}")
    return N, R, frames

# ----------------------------------------------------------------------
# 画图 & 动画
# ----------------------------------------------------------------------
def _auto_sizes_by_first_mass_or_default(first_frame_xyz, N, mass_hint=None):
    if mass_hint is not None and len(mass_hint) == N:
        m = np.asarray(mass_hint, dtype=np.float64)
        s = 10.0 * (m / max(m.mean(), 1e-12))
        return np.clip(s, 2, 60)
    return np.clip(8 * np.ones(N), 2, 60)

def animate(N, R, frames, dim3: bool, interval_ms: int, save_path: str | None,
            mass_hint=None, *, max_points: int = -1, use_blit: bool = False,
            max_frames: int = -1):
    # ===== 1) 帧数截断 (--frames) =====
    if max_frames is not None and isinstance(max_frames, int) and max_frames > 0:
        R = min(R, max_frames)
        frames = frames[:R]
    # ===== 2) 粒子下采样 (--max-points), 随机无放回抽样 (固定 seed 以便可重放) =====
    sel_idx = np.arange(N)
    if isinstance(max_points, int) and max_points > 0 and N > max_points:
        rng = np.random.default_rng(42)
        sel_idx = np.sort(rng.choice(N, size=max_points, replace=False))
        N_old, N = N, int(max_points)
        frames = frames[:, sel_idx, :]
        if mass_hint is not None and len(mass_hint) == N_old:
            mass_hint = np.asarray(mass_hint)[sel_idx]
        print(f"[visualize] downsample: N={N_old} -> {N} (--max-points={max_points})")

    fig = plt.figure(figsize=(9, 7))
    if dim3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111, aspect="equal")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(alpha=0.3)

    # 轴范围 (全局, 固定, 避免抖动)
    lo = np.nanmin(frames, axis=(0,1)); hi = np.nanmax(frames, axis=(0,1))
    span = max(1e-6, float((hi - lo).max()))
    c = 0.5 * (lo + hi)
    def pad(i): return c[i] - 0.6*span, c[i] + 0.6*span
    ax.set_xlim(*pad(0)); ax.set_ylim(*pad(1))
    if dim3: ax.set_zlim(*pad(2))

    # 颜色: 按粒子编号循环, 轨迹容易跟踪 → 强制 np.array(N,4) 跨 matplotlib 3.1/3.3/3.5 稳定
    cmap = plt.get_cmap("tab10" if N <= 10 else "rainbow")
    colors = np.array([cmap(i / max(1, N - 1))[:4] for i in range(N)], dtype=np.float32)
    assert colors.shape == (N, 4), f"bug: colors shape={colors.shape}, expected ({N},4)"
    sizes = np.asarray(_auto_sizes_by_first_mass_or_default(frames[0], N, mass_hint), dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full(N, float(sizes), dtype=np.float32)
    sizes = sizes.reshape(-1)
    if sizes.shape[0] != N:
        # 兜底 (mass_hint 尺寸错配等情形)
        sizes = np.full(N, 8.0, dtype=np.float32)

    if dim3:
        # NOTE: Ubuntu 20.04 matplotlib 3.3.x 的 3D scatter 在 edgecolors="none"
        # 时与 depthshade=True 组合会触发 _zalpha broadcast (0,4) vs (N,4) bug.
        # → depthshade=False + edgecolors="face" 跨版本稳定.
        sc = ax.scatter(frames[0, :, 0], frames[0, :, 1], frames[0, :, 2],
                        s=sizes, c=colors, alpha=0.85,
                        edgecolors="face", linewidths=0.0, depthshade=False)
    else:
        sc = ax.scatter(frames[0, :, 0], frames[0, :, 1],
                        s=sizes, c=colors, alpha=0.85, edgecolors="none")

    title = ax.set_title("")
    # blit 模式需要 title 也是返回的 artist 之一, 且不重画背景
    blit_flag = bool(use_blit)
    if blit_flag:
        print("[visualize] blit=True (incremental redraw)")

    def update(k):
        title.set_text(f"Frame {k:>4d} / {R - 1}   (N={N} particles)")
        if dim3:
            sc._offsets3d = (frames[k, :, 0], frames[k, :, 1], frames[k, :, 2])
        else:
            sc.set_offsets(frames[k, :, 0:2])
        return (sc, title)

    anim = FuncAnimation(fig, update, frames=R, interval=interval_ms,
                         blit=blit_flag, repeat=True)
    if save_path:
        sp = Path(save_path)
        fps = max(1, 1000 // interval_ms)
        try:
            if sp.suffix.lower() == ".gif":
                anim.save(save_path, writer="pillow", fps=fps, dpi=110)
            else:
                anim.save(save_path, fps=fps, dpi=110)
            sz = Path(save_path).stat().st_size
            print(f"[save] wrote {save_path}  ({sz:,} bytes)")
        except Exception as e:
            print(f"[save] failed: {e}; fallback: last-frame PNG")
            fig.savefig(str(sp.with_suffix(".png")), dpi=130)
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"[show] no display ({e}); saving last-frame PNG: nbody_last_frame.png")
            fig.savefig("nbody_last_frame.png", dpi=130)

# ----------------------------------------------------------------------
def _normalize_argv(argv):
    """
    把 "visualize.py demo 2body ..." 这种位置形式转成等价的 --demo 形式:
      sys.argv[1..N] = ["demo", "2body", "--out", "x.gif"]
                ->    ["--demo", "2body", "--out", "x.gif"]
    若第一个非 flag 参数不是 demo, 保持不动。
    """
    inp = list(argv)
    # 找到第一个不以 '-' 开头的参数（就是 trajectory）
    for i in range(len(inp)):
        tok = inp[i]
        if tok.startswith('-'):
            # 处理 "--option VALUE" 这种: 跳过 VALUE (如果下一个不开始于 -, 且这是"需要值"的 flag: 例如 --demo, --out, --fmt, --save, --view, --interval, --frames, --max-points, --demo-frames, --demo-dt)
            value_flags = {"--demo","--out","--fmt","--save","--view",
                           "--interval","--frames","--max-points",
                           "--demo-frames","--demo-dt","--block-size"}
            if tok.split('=',1)[0] in value_flags and '=' not in tok:
                if i+1 < len(inp):
                    i += 1
            continue
        if tok == "demo":
            # demo <MODE> 位置形式: 抽取后面的 MODE (需要存在)
            mode = inp[i+1] if (i+1 < len(inp) and not inp[i+1].startswith('-')) else None
            new = inp[:i]
            if mode is not None:
                new += ["--demo", mode]
                new += inp[i+2:]
            else:
                # demo 没跟 MODE, 还是保留原形式, 让 argparse 报错
                return argv
            return new
        else:
            # 第一个非 flag 是 trajectory, 不做任何变换
            return argv
    return argv

def main():
    # ---- 在 argparse 之前, 先把 "demo 2body" 位置形式归一化为 --demo ----
    raw_argv = _normalize_argv(sys.argv[1:])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("trajectory", nargs="?", default=None,
                    help="二进制或CSV轨迹文件 (根据 .csv 后缀自动识别, 也可显式 --csv)")
    ap.add_argument("--csv", action="store_true", help="强制用 CSV reader 读 trajectory")
    ap.add_argument("--demo", choices=["2body", "3body", "cluster"], help="演示模式 (等价于 `demo <mode>` 位置形式)")
    ap.add_argument("--demo-frames", type=int, default=240)
    ap.add_argument("--demo-dt", type=float, default=0.02)
    # 维度: --view {2d,3d} (新写法, README 推荐) + --2d/--3d (老写法, 向后兼容)
    dim = ap.add_mutually_exclusive_group()
    dim.add_argument("--view", choices=["2d","3d"], default=None, help="视图 (2d/3d), README 推荐写法")
    dim.add_argument("--2d", action="store_true", default=False, dest="dim2_old")
    dim.add_argument("--3d", action="store_true", default=False, dest="dim3")
    # 动画参数
    ap.add_argument("--interval", type=int, default=30, help="帧间隔 ms")
    ap.add_argument("--frames", type=int, default=-1, help="最多渲染前 N 帧 (避免长轨迹卡死)")
    ap.add_argument("--max-points", type=int, default=-1,
                    help="每帧最多画 M 个粒子 (大 N 可视化必加! 推荐 4000~10000); N>M 时随机下采样")
    ap.add_argument("--blit", action="store_true", help="开启动画 blit 增量刷新 (能快 2~3 倍, 但不兼容某些 backend)")
    # 保存: --save (旧) + --out (新) 等价; --fmt 指定 gif/mp4 优先级高于 out 文件后缀
    ap.add_argument("--save", type=str, default=None, help="保存路径 (.gif/.mp4); --out 的别名")
    ap.add_argument("--out",  type=str, default=None, help="保存路径; 与 --save 等价 (README 推荐写法)")
    ap.add_argument("--fmt",  choices=["gif","mp4"], default=None,
                    help="保存格式 gif/mp4; 如果省略则按 --out 后缀推断, 再省略默认 gif")
    args = ap.parse_args(raw_argv)

    # ---- 1) 维度最终值 ----
    dim3 = False
    if args.view == "3d":          dim3 = True
    elif args.view == "2d":        dim3 = False
    elif getattr(args, "dim3", False):    dim3 = True
    elif getattr(args, "dim2_old", False): dim3 = False
    else:                                    dim3 = False  # 默认 2D

    # ---- 2) 保存路径最终值 ----
    save_path = args.out or args.save or None
    if save_path is not None:
        suffix = Path(save_path).suffix.lower()
        if args.fmt == "gif" and suffix not in (".gif",):
            save_path = str(Path(save_path).with_suffix(".gif"))
        elif args.fmt == "mp4" and suffix not in (".mp4",):
            save_path = str(Path(save_path).with_suffix(".mp4"))
        elif not suffix and args.fmt is None:
            save_path = save_path + ".gif"   # 无后缀 → 默认 gif

    # ---- 3) 数据加载 ----
    if args.trajectory:
        force_csv = args.csv or args.trajectory.lower().endswith(".csv")
        if force_csv:
            N, R, frames = load_csv_trajectory(args.trajectory)
        else:
            N, R, frames = load_binary_trajectory(args.trajectory)
    elif args.demo:
        N, R, frames = leapfrog_demo(args.demo, R=args.demo_frames, dt=args.demo_dt)
    else:
        print("ERROR: 必须提供 trajectory.bin/.csv 或 demo <2body|3body|cluster> / --demo <mode>")
        ap.print_help()
        sys.exit(1)

    # 自动 mkdir 保存目录
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    animate(N, R, frames, dim3=dim3, interval_ms=args.interval, save_path=save_path,
            max_points=args.max_points, use_blit=args.blit, max_frames=args.frames)

if __name__ == "__main__":
    main()
