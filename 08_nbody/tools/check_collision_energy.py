#!/usr/bin/env python3
"""check_collision_energy.py —— 对撞初值的能量分项对账。

为什么需要这个脚本
------------------
`nbody_ic --preset cluster_collision` 会打印一个解析期望值
`E_total = 2*E_sub + E_orb`，供与 `nbody_sim` 实测的 E0 对照。
但首次对照时实测比解析值偏负约 2.6%，需要判断这是 bug 还是可解释的效应。

单看总能量判不了，因为它把四个来源混在一起：

  E_total = W_11 + W_22  (两个子球各自的自引力)
          + W_12         (两球之间的相互作用势能)
          + T_int        (子球内部的热运动动能)
          + T_orb        (两球相对运动的动能)

所以这里**逐项独立算出来**，每一项各自与解析值对账。哪一项对不上，
问题就在那一项，不必猜。

三个待检验的解析式
------------------
1. 单球自能（截断 Plummer）。教科书值 W = -3*pi*m^2/(32*a)，但采样器把 r
   截断在 20a，条件分布的矩因此略有位移。n_version1 已实测量化：截断后
   |W| 大 0.709%、E 负 1.065%。这里两个值都报，看实测落在哪个附近。

2. 两球相互作用势能 W_12。生成器用的是**质点近似** -G*m1*m2/d。
   但对两个 Plummer 球，存在闭式解
       W_12 = -G*m1*m2 / sqrt(d^2 + (a1+a2)^2)
   （Plummer 势 -GM/sqrt(r^2+a^2) 本身就是"带软化的质点势"，两球卷积后
   软化长度相加）。d=8a 时两式差 3.0% —— 不大，但既然有闭式就该用闭式。
   本脚本用实际粒子的双重求和去裁定哪个式子对。

3. 动能。T_orb = 0.5*mu*v_rel^2 是精确的（质心运动可严格分离），
   T_int 应为 -W_sub/2（位力平衡）。

用法
----
    python check_collision_energy.py <ic_file> --sep 8 --vfrac 0.7 [--imp 1.5]

注意 O(N^2) 双重求和：N=4096 时约 1.7e7 对，numpy 分块几秒即可；
N 大于约 3 万建议加 --sample 抽样（结论不变，噪声变大）。
"""

import argparse
import sys

import numpy as np

PI = np.pi


def load_bodies(path):
    """读初值文件（每行 x y z vx vy vz mass，# 为注释）。"""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0].split("//", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            rows.append([float(v) for v in parts])
    if not rows:
        sys.exit(f"error: no particles parsed from {path}")
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0:3], arr[:, 3:6], arr[:, 6]


def pair_potential(pos_a, m_a, pos_b, m_b, block=512):
    """两组粒子之间的相互作用势能 -G * sum_i sum_j m_i m_j / r_ij（G=1）。

    分块算而非一次性建 (Na, Nb) 矩阵：N=32768 时那是 8.6 GB。
    """
    total = 0.0
    for start in range(0, len(pos_a), block):
        chunk = pos_a[start:start + block]
        d = chunk[:, None, :] - pos_b[None, :, :]
        r = np.sqrt((d * d).sum(axis=2))
        # 两组不相交时 r 不会为 0；保险起见屏蔽零距离（重合粒子）。
        np.maximum(r, 1e-300, out=r)
        total += float((np.outer(m_a[start:start + block], m_b) / r).sum())
    return -total


def self_potential(pos, m, block=512):
    """一组粒子的自引力势能，每对只计一次（i<j）。"""
    total = 0.0
    n = len(pos)
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = pos[start:stop, None, :] - pos[None, :, :]
        r = np.sqrt((d * d).sum(axis=2))
        w = np.outer(m[start:stop], m) / np.maximum(r, 1e-300)
        # 只保留 j > i 的上三角部分
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        w = np.where(cols > rows, w, 0.0)
        total += float(w.sum())
    return -total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--sep", type=float, default=8.0, help="质心间距 d（单位 a）")
    ap.add_argument("--vfrac", type=float, default=0.7, help="v_rel / v_esc")
    ap.add_argument("--imp", type=float, default=1.5, help="碰撞参数 b（单位 a）")
    ap.add_argument("--a", type=float, default=1.0, help="Plummer 标长 a")
    ap.add_argument("--sample", type=int, default=0,
                    help="每球抽样这么多粒子（0=全用）。仅用于大 N 提速")
    args = ap.parse_args()

    pos, vel, mass = load_bodies(args.path)
    n = len(mass)

    # 按 x 的符号分成两球。生成器把两球质心放在 ±d/2，
    # 而单球 90% 的质量在 r<4a 内，d=8a 时用 x=0 分界是干净的。
    # （若 d 很小两球已重叠，这个划分就不成立——脚本会在下面报告不平衡。）
    left = pos[:, 0] < 0.0
    right = ~left
    n1, n2 = int(left.sum()), int(right.sum())

    if args.sample > 0:
        rng = np.random.default_rng(0)
        idx_l = np.flatnonzero(left)
        idx_r = np.flatnonzero(right)
        if len(idx_l) > args.sample:
            idx_l = rng.choice(idx_l, args.sample, replace=False)
        if len(idx_r) > args.sample:
            idx_r = rng.choice(idx_r, args.sample, replace=False)
        # 抽样会同时缩小质量总和，为保持物理量可比要把质量按比例放大。
        scale_l = mass[left].sum() / mass[idx_l].sum()
        scale_r = mass[right].sum() / mass[idx_r].sum()
        p1, m1 = pos[idx_l], mass[idx_l] * scale_l
        p2, m2 = pos[idx_r], mass[idx_r] * scale_r
        v1, v2 = vel[idx_l], vel[idx_r]
        print(f"  [sampled {len(idx_l)}+{len(idx_r)} of {n1}+{n2}; "
              f"masses rescaled to preserve totals]")
    else:
        p1, m1, v1 = pos[left], mass[left], vel[left]
        p2, m2, v2 = pos[right], mass[right], vel[right]

    M1, M2 = m1.sum(), m2.sum()
    M = M1 + M2
    d, a = args.sep, args.a

    # ---- 势能分项 ----
    W11 = self_potential(p1, m1)
    W22 = self_potential(p2, m2)
    W12 = pair_potential(p1, m1, p2, m2)

    # ---- 动能分项：先分离质心运动 ----
    # 每球质心速度
    vc1 = (m1[:, None] * v1).sum(axis=0) / M1
    vc2 = (m2[:, None] * v2).sum(axis=0) / M2
    v_rel_vec = vc2 - vc1
    v_rel_meas = float(np.sqrt((v_rel_vec ** 2).sum()))
    mu = M1 * M2 / M
    T_orb = 0.5 * mu * v_rel_meas ** 2
    # 内部动能：在各自质心系里算
    T1 = 0.5 * float((m1 * ((v1 - vc1) ** 2).sum(axis=1)).sum())
    T2 = 0.5 * float((m2 * ((v2 - vc2) ** 2).sum(axis=1)).sum())

    # ---- 解析预期 ----
    # 单球自能：教科书完整 Plummer 与截断 Plummer（r_cut=20a）
    w_full = -3.0 * PI * M1 * M1 / (32.0 * a)
    w_trunc = w_full * 1.00709          # n_version1 实测的截断位移
    e_sub_full = -3.0 * PI * M1 * M1 / (64.0 * a)
    e_sub_trunc = e_sub_full * 1.01065

    # 相互作用势能：质点近似 vs Plummer 闭式
    w12_point = -M1 * M2 / d
    w12_plummer = -M1 * M2 / np.sqrt(d * d + (2.0 * a) ** 2)

    v_esc = np.sqrt(2.0 * M / d)
    v_rel_expect = args.vfrac * v_esc

    def rel(meas, ref):
        return (meas - ref) / abs(ref) * 100.0 if ref else float("nan")

    print(f"file            : {args.path}")
    print(f"split           : N={n1}+{n2}  M1={M1:.9g}  M2={M2:.9g}")
    print(f"params          : d={d}  a={a}  vfrac={args.vfrac}  b={args.imp}")
    print()
    print("--- kinetic: centre-of-mass separation ---")
    print(f"v_rel measured  : {v_rel_meas:.9g}")
    print(f"v_rel expected  : {v_rel_expect:.9g}   "
          f"(v_esc={v_esc:.9g})   diff {rel(v_rel_meas, v_rel_expect):+.4f}%")
    print(f"T_orb           : {T_orb:.9g}")
    print(f"T_internal      : {T1:.9g} + {T2:.9g} = {T1 + T2:.9g}")
    print()
    print("--- self-gravity of each sphere ---")
    print(f"W11 measured    : {W11:.9g}")
    print(f"W22 measured    : {W22:.9g}")
    print(f"  vs full Plummer     {w_full:.9g}   "
          f"({rel(W11, w_full):+.3f}%, {rel(W22, w_full):+.3f}%)")
    print(f"  vs truncated 20a    {w_trunc:.9g}   "
          f"({rel(W11, w_trunc):+.3f}%, {rel(W22, w_trunc):+.3f}%)")
    print(f"virial 2T/|W|   : {2 * T1 / abs(W11):.6f}, {2 * T2 / abs(W22):.6f}"
          "   (should be ~1)")
    print()
    print("--- interaction energy between the two spheres ---")
    print("这一项裁定生成器该用哪个公式：")
    print(f"W12 measured    : {W12:.9g}")
    print(f"  point mass  -M1*M2/d              = {w12_point:.9g}   "
          f"({rel(W12, w12_point):+.3f}%)")
    print(f"  Plummer  -M1*M2/sqrt(d^2+(2a)^2)  = {w12_plummer:.9g}   "
          f"({rel(W12, w12_plummer):+.3f}%)")
    better = "Plummer closed form" if abs(rel(W12, w12_plummer)) < abs(
        rel(W12, w12_point)) else "point mass"
    print(f"  -> closer: {better}")
    print()
    print("--- total ---")
    e_meas = W11 + W22 + W12 + T1 + T2 + T_orb
    print(f"E measured      : {e_meas:.9g}")
    for label, e_sub, w12 in (
            ("full Plummer  + point mass", e_sub_full, w12_point),
            ("full Plummer  + Plummer W12", e_sub_full, w12_plummer),
            ("truncated 20a + point mass", e_sub_trunc, w12_point),
            ("truncated 20a + Plummer W12", e_sub_trunc, w12_plummer)):
        e_pred = 2.0 * e_sub + (0.5 * mu * v_rel_expect ** 2 + w12)
        print(f"  {label:<28}: {e_pred:.9g}   ({rel(e_meas, e_pred):+.3f}%)")
    print()
    print(f"sampling noise scale 1/sqrt(N_per_sphere) = "
          f"{100.0 / np.sqrt(min(n1, n2)):.3f}%")
    print("判据：残差应当在这个噪声尺度内，且换 --seed 时会变号/变幅。")


if __name__ == "__main__":
    main()
