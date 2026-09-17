#!/usr/bin/env python3
"""CLI, independent numerical oracle, serialization, and hostile input tests."""
import argparse
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tensor_io import write_tensor, read_weights
from reference import quantize, output_bytes, FP4, FP8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("executable")
    p.add_argument("--backend", default="cpu")
    args = p.parse_args()
    exe = str(Path(args.executable).resolve())
    np.seterr(over="ignore", under="ignore", invalid="ignore")
    rng = np.random.default_rng(821)
    cases = 0
    with tempfile.TemporaryDirectory(prefix="lp_tests_") as tmp:
        root = Path(tmp)
        inp, cfg, weights, output, log = [root / s for s in ("input.bin", "config.toml", "weights.bin", "output.bin", "log.json")]
        command = [exe, "run", "--input", str(inp), "--config", str(cfg), "--weights", str(weights),
                   "--output", str(output), "--log", str(log), "--backend", args.backend,
                   "--warmup", "0", "--iterations", "1", "--cpu-repeats", "1"]

        def call(cmd, success=True):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if (r.returncode == 0) != success:
                raise AssertionError(f"Unexpected exit {r.returncode}\n{r.stdout}\n{r.stderr}\n{cmd}")

        def case(a, fmt="nvfp4", mode="block", source="fp32", target="fp32", rounding="nearest", block=None, reload=False):
            nonlocal cases
            block = block or (16 if fmt == "nvfp4" else 32)
            cfg.write_text(f'format = "{fmt}"\nblock_size = {block}\nscale_mode = "{mode}"\n'
                           f'output_type = "{target}"\nrounding = "{rounding}"\nseed = 18446744073709551615\n'
                           'target_gpu = "test # quoted comment"\n')
            a = write_tensor(inp, a, source)
            call(command)
            expected_data, expected_scale, global_scale, restored = quantize(a, fmt, block, mode == "tensor", rounding == "stochastic", 2**64 - 1)
            actual = read_weights(weights)
            np.testing.assert_array_equal(actual["data"], expected_data, err_msg=f"{fmt} packed codes")
            np.testing.assert_array_equal(actual["scales"], expected_scale, err_msg=f"{fmt} scales")
            assert actual["global_scale"].tobytes() == global_scale.tobytes()
            assert output.read_bytes()[32:] == output_bytes(restored, target), (fmt, mode, source, target, rounding, block)
            metrics = json.loads(log.read_text())
            assert metrics["weight_file_bytes"] == weights.stat().st_size
            assert metrics["packed_data_bytes"] == a.shape[0] * ((a.shape[1] + 1) // 2 if fmt == "nvfp4" else a.shape[1])
            if reload:
                again = root / "reload.bin"
                call([exe, "decode", "--weights", str(weights), "--output", str(again), "--backend", args.backend,
                      "--warmup", "0", "--iterations", "1"])
                assert again.read_bytes() == output.read_bytes()
            cases += 1

        a = rng.normal(size=(5, 37)).astype(np.float32)
        a[0, :3] = [0, -0., 50]
        for fmt, mode, source, target, rounding in itertools.product(
                ["mxfp8", "nvfp4"], ["block", "tensor"], ["fp32", "fp16"],
                ["fp32", "fp16", "bf16"], ["nearest", "stochastic"]):
            case(a, fmt, mode, source, target, rounding, reload=source == "fp16" and rounding == "nearest")
        for fmt, block in itertools.product(["mxfp8", "nvfp4"], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
            case(rng.normal(size=(3, 1051)), fmt=fmt, block=block, rounding="stochastic")
        for shape in [(1, 1), (1, 2), (1, 15), (1, 16), (1, 17), (1, 31), (1, 32), (1, 33), (19, 1)]:
            for fmt in ["mxfp8", "nvfp4"]:
                case(np.zeros(shape, np.float32), fmt=fmt, reload=shape == (1, 1))
        for fmt in ["mxfp8", "nvfp4"]:
            tiny = np.array([[0, -0., 2**-149, -2**-149, 2**-127, -2**-126]], np.float32)
            case(tiny, fmt=fmt)
            huge = np.array([[np.finfo(np.float32).max, -np.finfo(np.float32).max, 1, -1]], np.float32)
            case(huge, fmt=fmt)
            case(np.full((1, 32), 1e20, np.float32), fmt=fmt, target="fp16")
            metrics = json.loads(log.read_text())
            assert metrics["nonfinite_output_count"] == 32 and metrics["mse"] is None

        # Exact grids and every midpoint, including their immediate FP32 neighbors.
        for fmt, table, multiplier in [("mxfp8", FP8, 1), ("nvfp4", FP4, 448)]:
            mid = (table[:-1] + table[1:]) / 2
            v = np.concatenate([table, mid, np.nextafter(mid, np.float32(0)),
                                np.nextafter(mid, np.float32(np.inf))]) * np.float32(multiplier)
            case(np.concatenate([v, -v]).reshape(1, -1), fmt=fmt, mode="tensor")

        # Each malformed input must fail before GPU allocation or output writes.
        case(a, reload=True)
        good_input, good_weights, good_config = inp.read_bytes(), weights.read_bytes(), cfg.read_text()
        for invalid in [good_input[:-1], good_input + b"x", b"badmagic" + good_input[8:],
                        good_input[:8] + struct.pack("<Q", 2**63) + good_input[16:],
                        good_input[:32] + struct.pack("<f", float("nan")) + good_input[36:],
                        good_input[:32] + struct.pack("<f", float("inf")) + good_input[36:]]:
            inp.write_bytes(invalid); call(command, False); cases += 1
        inp.write_bytes(good_input)
        for invalid in [good_config + 'typo = 1\n', good_config + 'format = "nvfp4"\n',
                        good_config.replace("block_size = 16", "block_size = 3"),
                        good_config.replace('rounding = "nearest"', 'rounding = "typo"')]:
            cfg.write_text(invalid); call(command, False); cases += 1
        cfg.write_text(good_config)
        for invalid in [good_weights[:-1], good_weights + b"x", good_weights[:80] + struct.pack("<f", 0) + good_weights[84:],
                        good_weights[:40] + struct.pack("<I", 2) + good_weights[44:], good_weights[:-1] + b"\xff"]:
            weights.write_bytes(invalid)
            call([exe, "decode", "--weights", str(weights), "--output", str(output), "--backend", args.backend], False)
            cases += 1
        weights.write_bytes(good_weights)
        call([exe, "decode", "--weights", str(weights), "--output", str(weights), "--backend", args.backend], False)
        cases += 1
    print(f"PASS: {cases} {args.backend} cases; independent codebook oracle, all modes, tails, reload, malformed files")


if __name__ == "__main__":
    main()
