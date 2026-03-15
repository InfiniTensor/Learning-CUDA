#!/usr/bin/env python3
"""将 PNG/JPG 图像转换为项目使用的 raw 格式。"""

import struct
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


def load_with_pillow(src: str, grayscale: bool):
    if Image is None:
        raise RuntimeError("Pillow is not installed")

    image = Image.open(src)
    if grayscale:
        image = image.convert("L")
        array = np.array(image)
        channels = 1
    else:
        image = image.convert("RGB")
        array = np.array(image)
        channels = 3
    return array, channels


def load_with_opencv(src: str, grayscale: bool):
    if cv2 is None:
        raise RuntimeError("OpenCV Python bindings are not installed")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(src, flag)
    if image is None:
        raise RuntimeError(f"Cannot load image: {src}")

    if grayscale:
        array = image
        channels = 1
    else:
        array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        channels = 3
    return array, channels


def load_image(src: str, grayscale: bool):
    loaders = [load_with_pillow, load_with_opencv]
    last_error = None
    for loader in loaders:
        try:
            return loader(src, grayscale)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load {src}: {last_error}")


def img2raw(src: str, dst: str, grayscale: bool = False):
    array, channels = load_image(src, grayscale)
    height, width = array.shape[:2]

    with open(dst, "wb") as handle:
        handle.write(struct.pack("<III", width, height, channels))
        handle.write(array.tobytes())

    print(f"Saved: {dst}  ({width}x{height} C={channels})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: img2raw.py <input.png> <output.raw> [--gray]")
        sys.exit(1)

    source = Path(sys.argv[1]).expanduser()
    target = Path(sys.argv[2]).expanduser()
    img2raw(str(source), str(target), "--gray" in sys.argv)
