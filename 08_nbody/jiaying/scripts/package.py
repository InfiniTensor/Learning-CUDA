#!/usr/bin/env python3
"""Package source, reproducible evidence and animations without build/large data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {".venv", "data", "output", "__pycache__", ".git"}


def main():
    candidates = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDE or part.startswith("build") for part in rel.parts[:-1]):
            continue
        if path.suffix in {".zip", ".pyc", ".nsys-rep", ".ncu-rep", ".qdstrm", ".sqlite"} or path.name == "SHA256.json":
            continue
        candidates.append(path)
    candidates.sort()
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
    manifest_path = ROOT / "reports/SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    destination = ROOT / "NBody_CUDA_jiaying.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in candidates + [manifest_path]:
            archive.write(path, Path("08_nbody/jiaying") / path.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for rel, digest in manifest.items():
            assert hashlib.sha256(archive.read("08_nbody/jiaying/" + rel)).hexdigest() == digest
    print(f"Verified {len(manifest)} files; {destination.stat().st_size:,} bytes; {destination}")


if __name__ == "__main__":
    main()
