"""交替复测本轮留存的 before/step2 二进制；在项目根目录执行。"""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
directory = root / "recheck"
directory.mkdir(exist_ok=False)
records = []
binary_hashes = {}
for index, n in enumerate((1 << 20, 4 << 20, 16 << 20)):
    for fmt in ("mxfp8", "nvfp4"):
        stages = ("before", "step2") if index % 2 == 0 else ("step2", "before")
        for stage in stages:
            binary = root / stage / ("pipeline_" + fmt)
            binary_hashes[str(binary.relative_to(root))] = hashlib.sha256(binary.read_bytes()).hexdigest()
            result = subprocess.run([str(binary), "--benchmark", str(n), "20"],
                                    text=True, capture_output=True, check=True)
            (directory / f"{stage}_{fmt}_{n}.jsonl").write_text(result.stdout)
            rows = [dict(json.loads(line), stage=stage) for line in result.stdout.splitlines()
                    if line.startswith("{")]
            records.extend(rows)
            row = next(r for r in rows if r["op"] == "quant_optimized" and r["scope"] == "resident_gpu")
            print(stage, fmt, n, row["median_ms"], flush=True)
(directory / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
(directory / "binary-sha256.json").write_text(json.dumps(binary_hashes, indent=2) + "\n")
