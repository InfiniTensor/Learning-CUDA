#!/usr/bin/env python3
"""Package sources, reproducible data and report, excluding local build artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = sorted(f for f in root.rglob("*") if f.is_file()
                   and not any(part.startswith("build") or part == "__pycache__" for part in f.relative_to(root).parts[:-1])
                   and f.suffix not in (".nsys-rep", ".ncu-rep", ".sqlite", ".qdstrm"))
    manifest = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
                if f.name != "manifest.json"}
    manifest_path = root / "results" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest_path not in files:
        files.append(manifest_path)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("05_option_pricing/jiaying") / f.relative_to(root))
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
    print(f"Packaged {len(files)} files: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
