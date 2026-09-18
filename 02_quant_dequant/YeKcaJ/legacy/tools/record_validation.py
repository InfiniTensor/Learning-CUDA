#!/usr/bin/env python3
"""汇集最近一次成功验证的日志与哈希，不改动冻结输入或 golden。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.evaluation).read_text())
    if len(records) != 144 or not all(r["cpu_quant_match"] and r["cpu_dequant_match"] for r in records):
        raise RuntimeError("evaluation must contain 144 passing comparisons")
    destination = Path(args.directory).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = dict(evaluation_cases=144, formats={})
    for fmt in ("MXFP8", "NVFP4"):
        source = ROOT / ("CUDA" + fmt) / "build/tests/results"
        ctest = (source / "ctest.log").read_text()
        if "100% tests passed, 0 tests failed out of 29" not in ctest:
            raise RuntimeError(f"CTest not passed: {fmt}")
        hashes = (source / "cpu_hashes.log").read_text()
        if hashes.count(": OK") != 15:
            raise RuntimeError(f"CPU hash verification incomplete: {fmt}")
        names = ["ctest.log", "cpu_hashes.log"]
        for variant in ("", "pipeline_"):
            for check in ("memcheck", "racecheck", "synccheck"):
                name = variant + check + ".log"
                text = (source / name).read_text()
                expected = "0 errors, 0 warnings" if check == "racecheck" else "ERROR SUMMARY: 0 errors"
                if expected not in text or (check == "memcheck" and "0 bytes leaked" not in text):
                    raise RuntimeError(f"sanitizer did not pass: {fmt}/{name}")
                names.append(name)
        target = destination / fmt; target.mkdir()
        for name in names:
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(ROOT / ("CUDA" + fmt) / "build/Testing/Temporary/LastTest.log", target / "LastTest.log")
        summary["formats"][fmt] = dict(ctest_passed=29, frozen_hashes=15, legacy_and_pipeline_sanitizers="PASS")
    manifest = {}
    for folder in ("Core", "CPUMXFP8", "CPUNVFP4", "CUDAMXFP8", "CUDANVFP4", "CUDACommon", "tools", "configs"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "build" in path.parts or "__pycache__" in path.parts or "results" in path.parts:
                continue
            if path.suffix in (".cpp", ".cu", ".cuh", ".h", ".py", ".sh", ".toml", ".fp32", ".mxfp8", ".nvfp4") or path.name == "CMakeLists.txt":
                manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in destination.rglob("*.log"):
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "SHA256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
