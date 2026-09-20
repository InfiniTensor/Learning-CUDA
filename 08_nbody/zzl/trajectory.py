"""Strict reader for int32 P,R followed by float32 xyz in (P,R,3) order."""
import json
from pathlib import Path
import struct
import sys
import numpy as np

FINITE_SCAN_BYTES = 1024 * 1024

def load_trajectory(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError("Trajectory header is truncated")
    particles, records = struct.unpack("<ii", header)
    if particles <= 0 or records <= 0:
        raise ValueError("P and R must both be positive")
    expected = 8 + particles * records * 3 * 4
    if path.stat().st_size != expected:
        raise ValueError(f"Wrong trajectory size: expected {expected}, got {path.stat().st_size}")
    metadata_path = Path(str(path) + ".json")
    metadata = {"steps": list(range(records)), "dt": 1.0, "time_is_record_index": True}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict):
            raise ValueError("Trajectory metadata must be a JSON object")
        if metadata.get("format") != "nbody-particle-major-v1" or metadata.get("endianness") != "little":
            raise ValueError("Unsupported trajectory metadata format")
        if metadata.get("dtype") != "float32":
            raise ValueError("Unsupported trajectory metadata dtype")
        if (type(metadata.get("particles")) is not int
                or type(metadata.get("records")) is not int
                or metadata["particles"] != particles or metadata["records"] != records):
            raise ValueError("Metadata shape disagrees with binary header")
        steps = metadata.get("steps")
        # Compare Python integers directly: fixed-width subtraction can wrap.
        if (not isinstance(steps, list) or len(steps) != records
                or any(type(step) is not int or not 0 <= step <= 2147483647 for step in steps)
                or steps[0] != 0
                or any(steps[i] <= steps[i - 1] for i in range(1, records))):
            raise ValueError("Metadata steps must start at 0 and strictly increase")
        dt = metadata.get("dt")
        if type(dt) not in (int, float) or not 0 < dt <= sys.float_info.max:
            raise ValueError("Metadata dt must be positive and finite")
    positions = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(particles, records, 3))
    # Bound scan memory by bytes, including a single particle with a long history.
    flat = positions.reshape(-1)
    values_per_scan = FINITE_SCAN_BYTES // positions.dtype.itemsize
    try:
        for base in range(0, flat.size, values_per_scan):
            if not np.isfinite(flat[base:base + values_per_scan]).all():
                raise ValueError("Trajectory contains NaN or infinity")
    except Exception:
        positions._mmap.close()
        raise
    return positions, metadata


def close_pairs(points, distance, max_pairs=2000):
    """3D spatial hash of one saved frame. Returns exact pairs until a declared cap.

    This is a proximity screen, not a physical collision predictor; no radii or
    between-frame motion are available in the specified trajectory format.
    """
    if not np.isfinite(distance) or distance <= 0 or max_pairs <= 0:
        raise ValueError("Positive finite distance and positive max_pairs required")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite (N,3) points")
    # Cell indices come from x/distance; an unrepresentable quotient would make
    # int(floor(...)) raise OverflowError instead of rejecting the threshold.
    if not np.isfinite(float(np.abs(points).max()) / distance):
        raise ValueError("distance is too small for the coordinate scale")
    cells = {}
    pairs = []
    distance2 = distance * distance
    for i, point in enumerate(points):
        # Python integers prevent overflow for widely separated normalized positions.
        cell = tuple(int(np.floor(x / distance)) for x in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for j in cells.get(key, ()):
                        delta = point - points[j]
                        squared = float(delta @ delta)
                        if squared < distance2:
                            if len(pairs) == max_pairs:
                                return pairs, True
                            pairs.append((j, i, float(np.sqrt(squared))))
        cells.setdefault(cell, []).append(i)
    return pairs, False
