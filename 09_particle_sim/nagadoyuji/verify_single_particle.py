#!/usr/bin/env python3
"""
Correctness check: single particle in uniform B should move in a helix.
Theory: omega = |q|*B/m, r_gyro = v_perp / omega.
Load trajectory from binary, compute numerical radius, compare to theory.
"""
import sys
import numpy as np

def load_trajectory(path):
    with open(path, "rb") as f:
        pp = np.frombuffer(f.read(4), dtype=np.int32)[0]
        rr = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.fromfile(f, dtype=np.float32, count=pp * rr * 3)
    return data.reshape(pp, rr, 3)

def load_particle_and_field(particles_path, field_path):
    with open(particles_path) as f:
        line = f.readline()
    parts = line.split()
    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    vx, vy, vz = float(parts[3]), float(parts[4]), float(parts[5])
    q, m = float(parts[6]), float(parts[7])
    with open(field_path) as f:
        line = f.readline()
    Bx, By, Bz = [float(x) for x in line.split()]
    return (x, y, z), (vx, vy, vz), q, m, (Bx, By, Bz)

def main():
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "out.bin"
    particles_path = sys.argv[2] if len(sys.argv) > 2 else "data/particles.txt"
    field_path = sys.argv[3] if len(sys.argv) > 3 else "data/field.txt"

    data = load_trajectory(traj_path)
    if data.shape[0] != 1:
        print("Expected single-particle trajectory; got", data.shape[0], "particles")
        sys.exit(1)
    pos = data[0]
    rr = pos.shape[0]

    (x0, y0, z0), (vx, vy, vz), q, m, (Bx, By, Bz) = load_particle_and_field(
        particles_path, field_path
    )
    B = np.sqrt(Bx*Bx + By*By + Bz*Bz)
    if B < 1e-30:
        print("B is zero, no gyration")
        sys.exit(0)
    v_parallel = (vx*Bx + vy*By + vz*Bz) / B
    v_perp = np.sqrt(max(0, vx*vx + vy*vy + vz*vz - v_parallel*v_parallel))
    omega_theory = abs(q) * B / m
    r_gyro_theory = v_perp / omega_theory if omega_theory > 1e-30 else 0.0

    cx, cy, cz = pos[:, 0].mean(), pos[:, 1].mean(), pos[:, 2].mean()
    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy
    dz = pos[:, 2] - cz
    dot = (dx*Bx + dy*By + dz*Bz) / (B * B)
    px = dx - dot * Bx
    py = dy - dot * By
    pz = dz - dot * Bz
    r_numerical = np.sqrt(px*px + py*py + pz*pz).mean()
    T_gyro = 2 * np.pi / omega_theory
    dt_max = T_gyro / 20.0

    print("--- Single particle in uniform B ---")
    print("Theory: omega = |q|*B/m =", omega_theory, "rad/s")
    print("        T_gyro = 2*pi/omega =", T_gyro, "s")
    print("        r_gyro = v_perp/omega =", r_gyro_theory, "m")
    print("Numerical mean gyro radius (in plane):", r_numerical, "m")
    err_r = abs(r_numerical - r_gyro_theory) / (r_gyro_theory + 1e-30)
    print("Relative error in radius:", err_r)
    if err_r < 0.2:
        print("PASS: radius within 20% of theory")
    else:
        print("CHECK: radius error large.")
        print("  Use dt < %.2e s (e.g. params_verify.txt)." % dt_max)
        print("  Re-run: ./particle_sim data/particles_1.txt data/field.txt data/params_verify.txt out.bin")
        print("  Then:  python verify_single_particle.py out.bin data/particles_1.txt data/field.txt")

if __name__ == "__main__":
    main()
