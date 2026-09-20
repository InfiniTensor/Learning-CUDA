#!/usr/bin/env python3
"""Package submission sources and measured evidence, excluding generated data."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root.parents[2] / "trade_planning_jiaying.zip")
    args = parser.parse_args()
    files = [root / name for name in ("CMakeLists.txt", ".gitignore", "README.md", "REPORT.md")]
    for folder in ("include", "src", "tests", "scripts", "reports"):
        files.extend(p for p in (root / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.name != "SHA256.json")
    files = sorted(files)
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = root / "reports/SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    files.append(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("06_trade_planning/jiaying") / path.relative_to(root))
        # Preserve the repository's license when distributing a standalone ZIP.
        license_path = root.parents[1] / "LICENSE"
        if license_path.exists():
            archive.write(license_path, "LICENSE")
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for relative, expected in manifest.items():
            actual = hashlib.sha256(archive.read("06_trade_planning/jiaying/" + relative)).hexdigest()
            assert actual == expected, relative
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"package": str(args.output.resolve()), "files": len(files),
                      "bytes": args.output.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
