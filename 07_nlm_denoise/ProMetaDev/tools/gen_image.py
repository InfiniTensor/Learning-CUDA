#!/usr/bin/env python3
"""生成 NLM 测试用的合成图像（干净图 + 加噪图）。

图像内容刻意包含 NLM 关注的四类结构：
  ① 大面积平坦/缓变区域（考验噪声抑制）
  ② 陨石坑式的圆形边缘（考验边缘保持）
  ③ 细裂隙状亮暗细线（考验细节保持）
  ④ 细颗粒纹理区（容易被过度平滑）

用法:
  python3 gen_image.py --size 1920x1080 --channels 3 --sigma 25 \
      --seed 1 --out-clean clean.png --out-noisy noisy.png
"""
import argparse
import numpy as np
from PIL import Image


def make_field(W, H, seed, scale=1.0):
    """低频平滑场 + 圆形结构 + 细线 + 纹理，返回 float 数组 (H, W) 取值约 0-255"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # ① 低频缓变底（几个正弦叠加）
    base = np.zeros((H, W), np.float64)
    for k in range(4):
        f = (0.6 + 1.7 * k) / max(W, H) * 12.0 * scale
        ax = rng.uniform(0, 2 * np.pi)
        ay = rng.uniform(0, 2 * np.pi)
        base += rng.uniform(0.4, 1.0) * np.sin(f * xx + ax) * np.cos(f * yy + ay)
    base = 128.0 + 45.0 * base / (np.abs(base).max() + 1e-9)

    img = base.copy()

    # ② 陨石坑：中心暗、边缘亮的圆环
    n_crater = max(3, int(W * H / 260000))
    for _ in range(n_crater):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.03, 0.12) * min(W, H)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inside = d < r
        rim = (d >= r * 0.85) & (d <= r * 1.02)
        img[inside] -= 42.0 * (1.0 - (d[inside] / r) ** 2)
        img[rim] += 55.0

    # ③ 细裂隙：随机走向的细亮/暗线
    n_crack = max(4, int(W * H / 220000))
    for _ in range(n_crack):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(0.08, 0.35) * max(W, H)
        width = rng.uniform(1.0, 2.2)
        t = np.linspace(0, length, int(length * 2) + 2)
        xc = x0 + t * np.cos(ang) + rng.normal(0, 1.2, t.size).cumsum() * 0.35
        yc = y0 + t * np.sin(ang) + rng.normal(0, 1.2, t.size).cumsum() * 0.35
        bright = rng.random() < 0.5
        for px, py in zip(xc, yc):
            ix, iy = int(px), int(py)
            if 0 <= ix < W and 0 <= iy < H:
                d = np.sqrt((xx[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] - px) ** 2 +
                            (yy[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] - py) ** 2)
                w = np.exp(-(d ** 2) / (2 * width ** 2))
                sl = (slice(max(0, iy - 2), iy + 3), slice(max(0, ix - 2), ix + 3))
                img[sl] += (60.0 if bright else -60.0) * w

    # ④ 细纹理区（棋盘 + 噪声颗粒），放在画面一角
    tw, th = W // 5, H // 5
    x1, y1 = W - tw - 1, H - th - 1
    yy2, xx2 = np.mgrid[0:th, 0:tw]
    checker = ((xx2 // 2 + yy2 // 2) % 2) * 55.0
    tex = 165.0 + checker + rng.normal(0, 6.0, (th, tw))
    img[y1:y1 + th, x1:x1 + tw] = tex

    return np.clip(img, 0.0, 255.0)


def add_noise(img, sigma, rng):
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0.0, 255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1920x1080", help="宽x高，如 1920x1080")
    ap.add_argument("--channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--sigma", type=float, default=25.0, help="高斯噪声标准差（0-255）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-clean", required=True)
    ap.add_argument("--out-noisy", required=True)
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    rng = np.random.default_rng(args.seed + 1000)

    if args.channels == 1:
        clean = make_field(W, H, args.seed)
    else:
        # 三个通道用不同 seed，模拟彩色场景
        clean = np.stack([make_field(W, H, args.seed + i * 7) for i in range(3)], axis=2)
        # 通道间轻微混合，避免完全独立
        clean = 0.7 * clean + 0.15 * np.roll(clean, 1, axis=2) + 0.15 * np.roll(clean, -1, axis=2)
        clean = np.clip(clean, 0.0, 255.0)

    noisy = add_noise(clean, args.sigma, rng)

    Image.fromarray(clean.round().astype(np.uint8), mode="L" if args.channels == 1 else "RGB") \
        .save(args.out_clean)
    Image.fromarray(noisy.round().astype(np.uint8), mode="L" if args.channels == 1 else "RGB") \
        .save(args.out_noisy)

    d = noisy.astype(np.float64) - clean.astype(np.float64)
    print(f"[gen] {W}x{H}x{args.channels} sigma={args.sigma}  "
          f"clean -> {args.out_clean}  noisy -> {args.out_noisy}")
    print(f"[gen] 实测噪声 MAE={np.abs(d).mean():.2f}  RMSE={np.sqrt((d**2).mean()):.2f}")


if __name__ == "__main__":
    main()
