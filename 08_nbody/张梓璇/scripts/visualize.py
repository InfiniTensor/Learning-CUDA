#!/usr/bin/env python3
"""读取 [record][particle][xyz] 布局的 trajectory.bin 并播放动画。"""

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def read_trajectory(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError("轨迹文件太短，读不到 particle_count 和 record_count")
        particle_count, record_count = struct.unpack("=ii", header)
        if particle_count <= 0 or record_count <= 0:
            raise ValueError("轨迹文件头中的粒子数和帧数必须为正")
        data = np.fromfile(file, dtype=np.float32)

    expected = record_count * particle_count * 3
    if data.size != expected:
        raise ValueError(
            f"轨迹数据长度不对：预计 {expected} 个 float，实际 {data.size} 个"
        )
    positions = data.reshape(record_count, particle_count, 3)
    if not np.isfinite(positions).all():
        raise ValueError("轨迹里出现了 NaN 或 Inf，模拟可能已经数值发散")
    return positions


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = high - low
    padding = max(span * 0.08, 1.0e-3)
    return low - padding, high + padding


def animate_2d(positions: np.ndarray, interval: int, point_size: float):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("N-body trajectory (XY)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*axis_limits(positions[:, :, 0]))
    ax.set_ylim(*axis_limits(positions[:, :, 1]))
    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=point_size)
    frame_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        scatter.set_offsets(positions[frame, :, :2])
        frame_text.set_text(f"frame {frame}/{len(positions) - 1}")
        return scatter, frame_text

    animation = FuncAnimation(
        fig, update, frames=len(positions), interval=interval, blit=True
    )
    return fig, animation


def animate_3d(positions: np.ndarray, interval: int, point_size: float):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("N-body trajectory (XYZ)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(*axis_limits(positions[:, :, 0]))
    ax.set_ylim(*axis_limits(positions[:, :, 1]))
    ax.set_zlim(*axis_limits(positions[:, :, 2]))
    scatter = ax.scatter(
        positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=point_size
    )

    def update(frame: int):
        # Matplotlib 的 3D scatter 目前通过这个属性更新三维坐标。
        scatter._offsets3d = (
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
        )
        ax.set_title(f"N-body trajectory (XYZ) — frame {frame}/{len(positions) - 1}")
        return (scatter,)

    animation = FuncAnimation(
        fig, update, frames=len(positions), interval=interval, blit=False
    )
    return fig, animation


def main() -> None:
    parser = argparse.ArgumentParser(description="播放 CUDA N-body 的二进制轨迹")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--3d", dest="three_d", action="store_true", help="使用 3D 动画")
    parser.add_argument("--interval", type=int, default=40, help="帧间隔，单位 ms")
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--save", type=Path, help="保存为 GIF 或 MP4；扩展名决定格式")
    parser.add_argument("--info", action="store_true", help="只检查并打印轨迹信息")
    args = parser.parse_args()

    positions = read_trajectory(args.trajectory)
    print(f"records={positions.shape[0]}, particles={positions.shape[1]}")
    print(f"coordinate_min={positions.min(axis=(0, 1))}")
    print(f"coordinate_max={positions.max(axis=(0, 1))}")
    if args.info:
        return

    fig, animation = (
        animate_3d(positions, args.interval, args.point_size)
        if args.three_d
        else animate_2d(positions, args.interval, args.point_size)
    )
    if args.save:
        animation.save(args.save)
        print(f"saved: {args.save.resolve()}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
