#!/usr/bin/env python3
"""Export stable CSV summaries directly from a Nsight Systems SQLite file."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(args.sqlite)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"}
    missing = required - tables
    if missing:
        raise SystemExit(f"missing Nsight tables: {sorted(missing)}")

    gpu = db.execute("select * from TARGET_INFO_GPU limit 1").fetchone()
    gpu_name = gpu["name"] if gpu else "unknown"
    kernel_rows = []
    query = """
        select s.value as kernel_name, count(*) as calls,
               sum(k.end-k.start) as total_ns, avg(k.end-k.start) as avg_ns,
               min(k.end-k.start) as min_ns, max(k.end-k.start) as max_ns,
               k.registersPerThread, k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ,
               k.staticSharedMemory, k.dynamicSharedMemory
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id=k.shortName
        group by s.value, k.registersPerThread, k.gridX, k.gridY, k.gridZ,
                 k.blockX, k.blockY, k.blockZ, k.staticSharedMemory, k.dynamicSharedMemory
        order by total_ns desc
    """
    for row in db.execute(query):
        threads = row["blockX"] * row["blockY"] * row["blockZ"]
        occupancy = ""
        if gpu and threads:
            warps_per_block = math.ceil(threads / gpu["threadsPerWarp"])
            limits = [gpu["maxBlocksPerSm"], gpu["maxWarpsPerSm"] // warps_per_block]
            if row["registersPerThread"]:
                limits.append(gpu["maxRegistersPerSm"] // (row["registersPerThread"] * threads))
            shared = row["staticSharedMemory"] + row["dynamicSharedMemory"]
            if shared:
                limits.append(gpu["maxShmemPerSm"] // shared)
            blocks = max(0, min(limits))
            occupancy = 100.0 * blocks * warps_per_block / gpu["maxWarpsPerSm"]
        kernel_rows.append({
            "gpu_name": gpu_name,
            "kernel_name": row["kernel_name"],
            "calls": row["calls"],
            "total_ms": f'{row["total_ns"] / 1e6:.9g}',
            "avg_us": f'{row["avg_ns"] / 1e3:.9g}',
            "min_us": f'{row["min_ns"] / 1e3:.9g}',
            "max_us": f'{row["max_ns"] / 1e3:.9g}',
            "registers_per_thread": row["registersPerThread"],
            "grid": f'{row["gridX"]}x{row["gridY"]}x{row["gridZ"]}',
            "block": f'{row["blockX"]}x{row["blockY"]}x{row["blockZ"]}',
            "static_shared_bytes": row["staticSharedMemory"],
            "dynamic_shared_bytes": row["dynamicSharedMemory"],
            "resource_limit_occupancy_pct": "" if occupancy == "" else f"{occupancy:.9g}",
        })
    write_csv(args.output_dir / "nsys_kernel_summary.csv", list(kernel_rows[0]), kernel_rows)

    api_rows = []
    api_query = """
        select s.value as api_name, count(*) as calls,
               sum(r.end-r.start) as total_ns, avg(r.end-r.start) as avg_ns,
               min(r.end-r.start) as min_ns, max(r.end-r.start) as max_ns
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on s.id=r.nameId
        group by s.value order by total_ns desc
    """
    for row in db.execute(api_query):
        api_rows.append({
            "api_name": row["api_name"],
            "calls": row["calls"],
            "total_ms": f'{row["total_ns"] / 1e6:.9g}',
            "avg_us": f'{row["avg_ns"] / 1e3:.9g}',
            "min_us": f'{row["min_ns"] / 1e3:.9g}',
            "max_us": f'{row["max_ns"] / 1e3:.9g}',
        })
    write_csv(args.output_dir / "nsys_cuda_api_summary.csv", list(api_rows[0]), api_rows)

    if gpu:
        device_row = {
            "gpu_name": gpu_name,
            "compute_capability": f'{gpu["computeMajor"]}.{gpu["computeMinor"]}',
            "sm_count": gpu["smCount"],
            "clock_rate_hz": gpu["clockRate"],
            "memory_bandwidth_bytes_per_s": gpu["memoryBandwidth"],
            "l2_cache_bytes": gpu["l2CacheSize"],
            "max_warps_per_sm": gpu["maxWarpsPerSm"],
            "max_blocks_per_sm": gpu["maxBlocksPerSm"],
            "max_registers_per_sm": gpu["maxRegistersPerSm"],
            "max_shared_memory_per_sm": gpu["maxShmemPerSm"],
        }
        write_csv(args.output_dir / "nsys_device_summary.csv", list(device_row), [device_row])
    print(f"wrote Nsight Systems summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
