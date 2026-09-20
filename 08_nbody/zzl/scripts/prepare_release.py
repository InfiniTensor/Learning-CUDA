"""Create source-and-binary provenance for a verified local release."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_sources(root: Path) -> list[Path]:
    sources = {root / name for name in
               ("CMakeLists.txt", "generate.py", "trajectory.py", "visualize.py")}
    for folder, suffixes in (("src", {".cpp", ".cu"}), ("include", {".hpp", ".h"}),
                             ("tests", {".py"}), ("scripts", {".py"})):
        sources.update(path for path in (root / folder).rglob("*")
                       if path.is_file() and path.suffix in suffixes)
    return sorted(sources)


def current_commit(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evidence" / "submission-artifact.json")
    args = parser.parse_args()
    executable = ROOT / "build-wsl-review" / "nbody"
    if not executable.is_file():
        parser.error(f"Missing verified executable: {executable}")
    sources = release_sources(ROOT)
    artifact = {
        "date": date.today().isoformat(),
        "source_commit": current_commit(ROOT),
        "executable_sha256": sha256(executable),
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "tests": "submission-tests.json",
        "benchmark": "submission-benchmark.json",
        "end_to_end": "submission-end-to-end.json",
        "platform": "WSL Ubuntu-24.04, CUDA 12.8, NVIDIA RTX 5060 Laptop",
        "limitations": [
            "Windows binary not rebuilt",
            "No domestic GPU backend",
            "Nsight Compute metrics unavailable; Nsight Systems CUDA trace is included",
        ],
        "source_note": "Source hashes and executable hash were captured after the recorded acceptance run.",
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
