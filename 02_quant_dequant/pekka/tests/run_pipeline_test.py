#!/usr/bin/env python3
"""Run two small GPU pipelines and validate their externally visible files."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    executable = Path(sys.argv[1])
    smoke_test = Path(sys.argv[2])
    generator = Path(sys.argv[3])
    source_dir = Path(sys.argv[4])
    smoke = subprocess.run([str(smoke_test)], check=False)
    if smoke.returncode == 77:
        print("GPU pipeline skipped because no compatible CUDA device is visible")
        raise SystemExit(77)
    if smoke.returncode != 0:
        raise RuntimeError(f"CUDA smoke test failed with exit code {smoke.returncode}")
    with tempfile.TemporaryDirectory(prefix="qd_pipeline_") as directory:
        root = Path(directory)
        input_path = root / "input.qdat"
        subprocess.run(
            [sys.executable, str(generator), "generate", "--output", str(input_path),
             "--kind", "normal", "--rows", "3", "--cols", "11", "--dtype", "fp32"],
            check=True,
        )
        cases = (
            ("mxfp8_block", "mxfp8", "block", "fp16"),
            ("nvfp4_block", "nvfp4", "block", "fp16"),
            ("mxfp8_tensor", "mxfp8", "tensor", "fp32"),
            ("nvfp4_tensor", "nvfp4", "tensor", "bf16"),
            ("mxfp8_stochastic", "mxfp8", "block", "fp32"),
            ("nvfp4_stochastic", "nvfp4", "block", "fp32"),
        )
        for config_name, format_name, scale_mode, output_dtype in cases:
            qlow = root / f"{config_name}.qlow"
            output = root / f"{config_name}.qdat"
            metrics = root / f"{config_name}.json"
            subprocess.run(
                [str(executable), "--input", str(input_path), "--config",
                 str(source_dir / "configs" / f"{config_name}.toml"),
                 "--quantized", str(qlow), "--output", str(output),
                 "--metrics", str(metrics), "--warmup", "1", "--iterations", "2"],
                check=True,
            )
            result = json.loads(metrics.read_text(encoding="utf-8"))
            assert result["format"] == format_name
            assert result["scale_mode"] == scale_mode
            assert result["output_dtype"] == output_dtype
            assert result["elements"] == 33
            assert result["max_abs_error"] >= 0
            subprocess.run([sys.executable, str(generator), "inspect", str(output)], check=True)


if __name__ == "__main__":
    main()
