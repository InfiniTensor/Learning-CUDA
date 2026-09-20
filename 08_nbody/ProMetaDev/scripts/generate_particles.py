#!/usr/bin/env python3
"""生成 N 体初始条件 particles.txt（可直接给 CUDA NBody 程序使用）。

支持 3 种模型（题面常用场景 + 进阶 65536 粒子普朗特球）:
  - plummer   : Plummer 球 (恒星/暗物质晕经典模型, 稳定不易坍缩)
  - sphere    : 均匀质量球 + 切向位力速度 (冷球升级版, 能量更稳定)
  - disk      : 薄星系盘 + 开普勒转动 (用于可视化盘结构)

示例:
  python3 scripts/generate_particles.py plummer 65536 data/particles_65536.txt --seed 2026
  python3 scripts/generate_particles.py sphere  4096  data/particles_4096_sphere.txt --eps 0.02 --Rscale 3.0
  python3 scripts/generate_particles.py disk    8192  data/particles_disk.txt
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np

def plummer_sphere(N: int, seed: int, Mtot: float, Rscale: float):
    """经典 Plummer (1911) 密度分布 ρ ∝ (1 + r²/a²)^(-5/2)，能量/动量已自动归零，配合 G=1 数值稳定。"""
    rng = np.random.default_rng(seed)
    # 位置: 拒绝采样 or 反变换 (Aarseth 公式)
    #   半径 CDF: m = M*(r²/(r²+a²))^(3/2) → r = a / sqrt( m^(-2/3) - 1 )
    a = Rscale
    m = rng.uniform(1e-12, 1.0-1e-12, size=N)
    r = a / np.sqrt(np.power(m, -2/3) - 1.0)
    pts = rng.normal(size=(N, 3))
    rs = np.linalg.norm(pts, axis=1, keepdims=True)
    rs = np.where(rs == 0, 1, rs)
    X = (pts / rs) * r[:, None]                                                         # (N,3)
    # 速度: Aarseth plummer 速度抽样 (q 是均匀分数)
    v_esc = np.sqrt(2.0 * Mtot * G) / np.power(r * r + a * a, 0.25)   # escape speed at r
    q = rng.uniform(0.0, 0.1, size=N)                               # narrow bound support
    g = lambda q: q * (1 - q*q) ** 3.5                               # Aarseth g(q) peak shape accept
    # 简单接受-拒绝
    for i in range(N):
        while True:
            qq = rng.uniform(0.0, 1.0)
            yy = rng.uniform(0.0, 0.1)                                 # peak ~g(0.25)=0.1
            if yy <= qq * ((1 - qq*qq) ** 3.5):
                q[i] = qq
                break
    v_mag = q * v_esc
    vpts = rng.normal(size=(N, 3))
    vrs = np.linalg.norm(vpts, axis=1, keepdims=True)
    vrs = np.where(vrs == 0, 1, vrs)
    V = (vpts / vrs) * v_mag[:, None]
    return X, V

def uniform_sphere_with_virial_rotation(N: int, seed: int, Mtot: float, Rscale: float, eps_r: float):
    """均匀球 + 近似位力平衡切向速度，避免冷球坍缩大能量漂移。"""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    rs = np.linalg.norm(pts, axis=1, keepdims=True)
    rs = np.where(rs == 0, 1, rs)
    u = rng.uniform(0, 1, size=(N, 1))
    r = (u ** (1/3)) * Rscale
    X = (pts / rs) * r
    # 包含包围质量 M(<r) = Mtot * (r/Rscale)^3; 圆周 v_c = sqrt(G M(<r) / r)
    Menc = Mtot * ((r.flatten() / Rscale) ** 3)
    vc = np.sqrt(G * Menc / np.maximum(r.flatten(), eps_r))
    # 随机切向方向: 构造任意垂直于 X 的向量
    e_z = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(X, e_z)
    t1n = np.linalg.norm(t1, axis=1, keepdims=True)
    mask = (t1n < 1e-14).flatten()
    t1[mask] = np.array([1.0, 0.0, 0.0])
    t1n = np.where(t1n == 0, 1, t1n)
    t1 = t1 / t1n
    t2 = np.cross(X, t1)
    t2 = t2 / np.linalg.norm(t2, axis=1, keepdims=True)
    # 随机相位 phi: v = cos*vc*t1 + sin*vc*t2 (保留少量径向 0.1 小弥散)
    phi = rng.uniform(0, 2*np.pi, N)
    V = vc[:, None] * (np.cos(phi)[:, None] * t1 + np.sin(phi)[:, None] * t2)
    # 10% 速度弥散
    V += 0.1 * vc[:, None] * rng.normal(size=(N, 3))
    return X, V

def kepler_disk(N: int, seed: int, Mtot: float, Rscale: float):
    """薄盘 + 开普勒转动。"""
    rng = np.random.default_rng(seed)
    r = np.sort(rng.uniform(0.3 * Rscale, 3.0 * Rscale, N))
    phi = rng.uniform(0, 2 * np.pi, N)
    z = rng.normal(0, 0.03 * Rscale, N)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # 开普勒 v = sqrt(G*M_central / r), 中心占 0.95Mtot + 剩余0.05均匀分布
    Mc = 0.95 * Mtot
    v = np.sqrt(G * Mc / r)
    V = np.column_stack([-v * np.sin(phi), v * np.cos(phi), np.zeros(N)])
    return X, V

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", choices=["plummer", "sphere", "disk"])
    ap.add_argument("N", type=int, help="粒子总数")
    ap.add_argument("output", help="输出 particles.txt 路径")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--G", type=float, default=1.0, help="用于生成速度的 G (仅参考, 后续模拟可独立设置)")
    ap.add_argument("--Mtot", type=float, default=1.0, help="总质量 (N 颗均分, 总和=Mtot)")
    ap.add_argument("--Rscale", type=float, default=1.0, help="长度尺度")
    ap.add_argument("--eps", type=float, default=0.0, help="仅 sphere 模型: 软化下界防 0 除")
    args = ap.parse_args()

    global G
    G = args.G
    N = args.N
    if args.model == "plummer":
        X, V = plummer_sphere(N, args.seed, args.Mtot, args.Rscale)
    elif args.model == "sphere":
        X, V = uniform_sphere_with_virial_rotation(N, args.seed, args.Mtot, args.Rscale, max(args.eps, 1e-6))
    else:
        X, V = kepler_disk(N, args.seed, args.Mtot, args.Rscale)

    mass = np.full(N, args.Mtot / N, dtype=np.float64)

    # 归零 CM / 总动量 (标准做法, 避免整体平移/转动产生假的能量漂移)
    X -= (mass[:, None] * X).sum(axis=0) / mass.sum()
    V -= (mass[:, None] * V).sum(axis=0) / mass.sum()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# generated: model={args.model} N={N} seed={args.seed} "
                f"G={args.G} Mtot={args.Mtot} Rscale={args.Rscale}\n")
        for i in range(N):
            f.write(f"{X[i,0]: .10e} {X[i,1]: .10e} {X[i,2]: .10e}   "
                    f"{V[i,0]: .10e} {V[i,1]: .10e} {V[i,2]: .10e}   "
                    f"{mass[i]: .10e}\n")
    print(f"[gen] wrote {args.output}: N={N}, model={args.model}")
    print(f"      CM pos= [{X[:,0].mean():.3e}, {X[:,1].mean():.3e}, {X[:,2].mean():.3e}]")
    print(f"      |V| range=[{np.linalg.norm(V,axis=1).min():.3e}, {np.linalg.norm(V,axis=1).max():.3e}]")

if __name__ == "__main__":
    main()
