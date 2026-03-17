#!/usr/bin/env python3
"""
Read binary trajectory file from particle_sim and plot 3D animated trajectories.
Usage: python visualize.py <trajectory.bin> [--tail N] [--every K]
  --tail N: draw trail of last N frames per particle (default 20)
  --every K: show only every K-th particle to reduce clutter (default 1)
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def load_trajectory(path):
    with open(path, "rb") as f:
        pp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        rr = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.fromfile(f, dtype=np.float32, count=pp * rr * 3)
    data = data.reshape(pp, rr, 3)
    return data  # shape (num_particles, num_records, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_file", nargs="?", default="out.bin", help="Binary trajectory file")
    ap.add_argument("--tail", type=int, default=20, help="Trail length (frames)")
    ap.add_argument("--every", type=int, default=1, help="Plot every K-th particle")
    args = ap.parse_args()

    data = load_trajectory(args.traj_file)
    pp, rr, _ = data.shape
    print(f"Particles: {pp}, Record points: {rr}")

    indices = np.arange(0, pp, args.every)
    data_plot = data[indices]
    n_show = len(indices)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    tail = min(args.tail, rr)
    pts = ax.scatter([], [], [], c="b", s=5, alpha=0.8)
    lines = [ax.plot([], [], [], "b-", alpha=0.4, linewidth=0.5)[0] for _ in range(n_show)]

    def bounds():
        mn = data_plot.min(axis=(0, 1))
        mx = data_plot.max(axis=(0, 1))
        margin = max((mx - mn).max() * 0.1, 1e-10)
        return mn - margin, mx + margin

    mn, mx = bounds()
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Particle trajectories (3D)")

    def animate(frame):
        f = frame % rr
        x = data_plot[:, f, 0]
        y = data_plot[:, f, 1]
        z = data_plot[:, f, 2]
        pts._offsets3d = (x, y, z)
        start = max(0, f - tail)
        for i in range(n_show):
            lines[i].set_data(
                data_plot[i, start : f + 1, 0],
                data_plot[i, start : f + 1, 1],
            )
            lines[i].set_3d_properties(data_plot[i, start : f + 1, 2])
        return [pts] + lines

    anim = animation.FuncAnimation(
        fig, animate, frames=rr, interval=50, blit=False, repeat=True
    )
    plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
