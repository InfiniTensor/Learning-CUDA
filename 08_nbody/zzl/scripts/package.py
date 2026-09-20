"""Build a reviewed delivery archive from an explicit allowlist, with file hashes."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def write_member(archive, name, data):
    """Preserve portable Unix permissions even when packaging on Windows."""
    member = zipfile.ZipInfo("nbody_cuda/" + name)
    member.create_system = 3
    member.external_attr = (0o100755 if name == "bin/nbody" else 0o100644) << 16
    member.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(member, data, compresslevel=6)


def verify_output_paths(output):
    """Preserve the archive and both companion records as one release set."""
    for path in (output, output.with_suffix(".zip.sha256"),
                 output.with_suffix(".zip.verification.json")):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Choose a new archive name; existing release file: {path}")


def verify_release(root):
    """Reject stale or incomplete acceptance evidence before creating an archive."""
    def load(name):
        return json.loads((root / "evidence" / name).read_text(encoding="utf-8-sig"))

    artifact = load("submission-artifact.json")
    sources = artifact.get("source_sha256", {})
    required = {"CMakeLists.txt", "generate.py", "trajectory.py", "visualize.py"}
    for folder, suffixes in (("src", {".cpp", ".cu"}), ("include", {".hpp", ".h"}),
                             ("tests", {".py"}), ("scripts", {".py"})):
        required.update(p.relative_to(root).as_posix() for p in (root / folder).rglob("*")
                        if p.is_file() and p.suffix in suffixes)
    missing = required - sources.keys()
    if missing:
        raise ValueError("Acceptance source hashes missing: " + ", ".join(sorted(missing)))
    for name in sorted(required):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != sources[name]:
            raise ValueError("Source changed since acceptance: " + name)
    executable = hashlib.sha256((root / "build-wsl-review/nbody").read_bytes()).hexdigest()
    if executable != artifact.get("executable_sha256"):
        raise ValueError("Executable changed since acceptance")
    benchmark = load("submission-benchmark.json")
    if executable != benchmark.get("executable_sha256"):
        raise ValueError("Benchmark belongs to a different executable")
    tests = load("submission-tests.json")
    if (type(tests.get("tests_run")) is not int or tests["tests_run"] <= 0
            or any(type(tests.get(key)) is not int or tests[key] != 0
                   for key in ("failures", "errors", "skipped"))
            or tests.get("cuda_available") is not True):
        raise ValueError("A successful CUDA acceptance report without skips is required")
    if tests.get("executable_sha256") != artifact.get("executable_sha256"):
        raise ValueError("Acceptance report belongs to a different executable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "delivery" / "nbody_cuda_v1.0.zip")
    args = parser.parse_args()
    try:
        verify_output_paths(args.output)
        verify_release(ROOT)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    files = []
    for folder in ("src", "include", "tests", "examples", "outputs/demos"):
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix not in {".pyc", ".tmp"}]
    for name in ("README.md", "REPORT.md", "SUBMISSION.md", "CMakeLists.txt", "requirements.txt", "requirements-lock.txt",
                 "docs/PROFILING.md", "evidence/profile_inputs/five_steps.cfg",
                 "generate.py", "trajectory.py", "visualize.py", ".gitignore", ".clang-format",
                 "build-wsl-review/nbody", "scripts/package.py", "scripts/report.py",
                 "scripts/verify_nsys.py", "scripts/profile_nsys.py", "scripts/prepare_release.py",
                 "scripts/benchmark.py", "scripts/benchmark_end_to_end.py", "scripts/make_demos.py",
                 "evidence/submission-tests.json", "evidence/submission-tests.txt",
                 "evidence/submission-tool-tests.txt",
                 "evidence/submission-benchmark.json", "evidence/submission-end-to-end.json",
                 "evidence/submission-artifact.json", "evidence/cpu.json",
                 "evidence/submission-clean-build.txt",
                 "evidence/submission-audit-current.json", "evidence/submission-audit-current.txt",
                 "evidence/profiling-nsys.json", "evidence/nsys-2026-short.txt",
                 "evidence/nsys-2026-stats.txt", "evidence/nsys-2026-naive.txt",
                 "evidence/nsys-2026-naive-stats.txt",
                 "evidence/memcheck.txt", "evidence/synccheck.txt", "evidence/ncu.txt",
                 "evidence/profiling-status.json",
                 "outputs/submission-benchmark/baseline_4096.bin",
                 "outputs/submission-benchmark/baseline_4096.bin.json",
                 "outputs/submission-benchmark/cluster_4096.txt",
                 "outputs/submission-benchmark/steps_1000.cfg"):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    files = sorted(set(files))
    def archive_path(p):
        name = p.relative_to(ROOT).as_posix()
        return "bin/nbody" if name == "build-wsl-review/nbody" else name

    def payload(p):
        data = p.read_bytes()
        # Relocate recorded command paths for a portable submission; keep measurements intact.
        if p.suffix in {".json", ".txt", ".md"}:
            text = data.decode("utf-8-sig")
            native = ROOT.as_posix()
            linux = "/mnt/" + native[0].lower() + native[2:]
            text = text.replace(linux + "/", "./").replace(native + "/", "./")
            if p.name == "ncu.txt":
                text = re.sub(r"\(C:\\[^\r\n]*\\build\\nbody\.exe\)",
                              "(build/nbody.exe)", text)
            data = text.encode("utf-8")
        return data

    entries = [{"path": archive_path(p), "bytes": len(payload(p)),
                "sha256": hashlib.sha256(payload(p)).hexdigest()} for p in files]
    manifest = {"format": "nbody-delivery-manifest-v1", "files": entries,
                "path_note": "Machine-specific command paths are relocated; profiler executable path is anonymized. Measured values and error messages are retained.",
                "large_trajectory_note": "Regenerate the 65536-particle trajectory with scripts/benchmark.py; its measured shape and SHA256 are in evidence/submission-benchmark.json."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            write_member(archive, archive_path(path), payload(path))
        write_member(archive, "MANIFEST.json", json.dumps(manifest, indent=2))
    # Verify both ZIP CRC and every uncompressed file's SHA256, without executing anything.
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for entry in entries:
            payload = archive.read("nbody_cuda/" + entry["path"])
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise RuntimeError("Package hash mismatch: " + entry["path"])
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n", encoding="ascii")
    verification = {
        "archive": args.output.name,
        "sha256": checksum,
        "payload_files": len(files),
        "crc_verified": True,
        "payload_hashes_verified": True,
        "validation_scope": "Archive CRC and payload hashes; execution tests have separate records.",
    }
    args.output.with_suffix(".zip.verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.output), "files": len(files), "bytes": args.output.stat().st_size,
                      "sha256": checksum, "crc_and_hashes_verified": True}, indent=2))


if __name__ == "__main__":
    main()
