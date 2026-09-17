#!/usr/bin/env python3
"""Create a submission archive containing source and compact measured evidence."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="MXFP8_NVFP4_jiaying.zip")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    excluded = {"build", "build-cpu", "output", "data", "__pycache__", ".git"}
    files = [f for f in root.rglob("*") if f.is_file() and not any(
        part in excluded or part.startswith("build-") for part in f.relative_to(root).parts)
        and f.suffix not in {".zip", ".ncu-rep", ".nsys-rep", ".sqlite", ".qdstrm", ".pyc"}]
    prefix = f"02_quant_dequant/{root.name}/"
    manifest = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            rel = prefix + f.relative_to(root).as_posix()
            archive.write(f, rel)
            manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
        license_file = root.parents[1] / "LICENSE"
        if license_file.is_file(): archive.write(license_file, "LICENSE")
        archive.writestr("SHA256SUMS.txt", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad: raise RuntimeError(f"Archive verification failed: {bad}")
    print(f"Verified {len(files)} project files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
