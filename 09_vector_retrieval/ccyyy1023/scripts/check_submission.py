#!/usr/bin/env python3
"""Static submission check that does not require CUDA."""

import json
import pathlib
import sys


REQUIRED = [
    "CMakeLists.txt",
    "README.md",
    "REPORT.md",
    "include/gvs/search.hpp",
    "include/gvs/io.hpp",
    "src/cpu_search.cpp",
    "src/gpu_search.cu",
    "src/ivf_flat.cpp",
    "src/file_io.cpp",
    "src/main.cpp",
    "tests/test_search.cpp",
    "configs/exact.conf",
    "configs/ivf_flat.conf",
    "results/SUMMARY.md",
    "results/environment.txt",
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        print("missing files:", *missing, sep="\n- ", file=sys.stderr)
        return 1
    logs = [json.loads(path.read_text(encoding="utf-8"))
            for path in (root / "results").glob("*.json")]
    if not logs or not all(item.get("status") == "measured" for item in logs):
        print("all submitted JSON logs must be real measured runs", file=sys.stderr)
        return 1
    forbidden = ["TODO: fake", "RTX 4090: 999", "speedup: 100x"]
    text = "\n".join((root / name).read_text(encoding="utf-8", errors="replace")
                     for name in REQUIRED if name.endswith((".md", ".cpp", ".cu", ".hpp")))
    if any(token in text for token in forbidden):
        print("found forbidden fabricated marker", file=sys.stderr)
        return 1
    print(f"submission structure OK ({len(REQUIRED)} required files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
