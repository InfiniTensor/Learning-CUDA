"""Reader for the task's strict little-endian particle-major binary format."""
import json
from pathlib import Path
import numpy as np


def load_trajectory(path):
    path = Path(path)
    header = np.fromfile(path, dtype="<i4", count=2)
    if len(header) != 2 or np.any(header <= 0):
        raise ValueError("Invalid P,R header")
    particles, records = map(int, header)
    expected = 8 + particles * records * 3 * 4
    if path.stat().st_size != expected:
        raise ValueError(f"Expected {expected} bytes; got {path.stat().st_size}")
    points = np.memmap(path, dtype="<f4", offset=8, mode="r", shape=(particles, records, 3))
    sidecar = Path(str(path) + ".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {"steps": list(range(records)), "dt": 1, "time_is_record_index": True}
    steps = np.asarray(metadata["steps"])
    dt = metadata["dt"]
    if steps.shape != (records,) or not np.all(np.isfinite(steps)) or not np.isfinite(dt) or dt <= 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Invalid time metadata")
    return points, steps * dt, metadata


def close_pairs(points, radius):
    """Exact 3D threshold pairs in one recorded frame using a spatial hash.

    This reports proximity only, not continuous collision detection or prediction.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not np.all(np.isfinite(points)):
        raise ValueError("Non-finite positions")
    cells = {}
    marked = np.zeros(len(points), dtype=bool)
    count = 0
    for i, p in enumerate(points):
        key = tuple(np.floor(p.astype(np.float64) / radius).astype(np.int64))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        delta = p.astype(np.float64) - points[j]
                        if delta @ delta < radius * radius:
                            marked[i] = marked[j] = True
                            count += 1
        cells.setdefault(key, []).append(i)
    return marked, count
