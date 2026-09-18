"""交替复测严格除法修改前后，保留原始 JSONL 和二进制 SHA256。"""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
directory = root / "recheck"
directory.mkdir(exist_ok=False)
rows = []
hashes = {}
for index, n in enumerate((1 << 20, 4 << 20, 16 << 20)):
    for fmt in ("mxfp8", "nvfp4"):
        stages = ("before", "after-divide") if index % 2 == 0 else ("after-divide", "before")
        for stage in stages:
            binary = root / stage / ("pipeline_" + fmt)
            hashes[str(binary.relative_to(root))] = hashlib.sha256(binary.read_bytes()).hexdigest()
            result = subprocess.run([str(binary), "--benchmark", str(n), "20"],
                                    text=True, capture_output=True, check=True)
            (directory / f"{stage}_{fmt}_{n}.jsonl").write_text(result.stdout)
            values = [dict(json.loads(line), stage=stage) for line in result.stdout.splitlines()
                      if line.startswith("{")]
            rows.extend(values)
            row = next(x for x in values if x["op"] == "quant_optimized" and x["scope"] == "resident_gpu")
            print(stage, fmt, n, row["median_ms"], row["p95_ms"], flush=True)
(directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
(directory / "binary-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
