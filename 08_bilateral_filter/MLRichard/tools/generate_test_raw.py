#!/usr/bin/env python3
"""Generate a deterministic raw image for benchmark and profiling."""

import argparse
import struct
from pathlib import Path

import numpy as np


def build_image(width: int, height: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base_r = 255.0 * xx
    base_g = 255.0 * yy
    radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    base_b = np.clip(255.0 * (1.0 - 1.5 * radial), 0.0, 255.0)
    checker = (((np.floor(xx * 32) + np.floor(yy * 32)) % 2) * 32.0).astype(np.float32)
    noise = rng.normal(0.0, 10.0, size=(height, width, channels)).astype(np.float32)

    if channels == 1:
        gray = 0.45 * base_r + 0.35 * base_g + 0.20 * base_b + checker
        image = gray[..., None]
    else:
        image = np.stack([base_r + checker, base_g, base_b + 0.5 * checker], axis=-1)

    image = np.clip(image + noise, 0.0, 255.0).astype(np.uint8)
    return image if channels > 1 else image[..., 0]


def write_raw(path: Path, array: np.ndarray, channels: int) -> None:
    height, width = array.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", width, height, channels))
        handle.write(array.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic raw image input")
    parser.add_argument("output", type=Path, help="output raw path")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    image = build_image(args.width, args.height, args.channels, args.seed)
    write_raw(args.output, image, args.channels)
    print(f"Saved: {args.output} ({args.width}x{args.height} C={args.channels})")
