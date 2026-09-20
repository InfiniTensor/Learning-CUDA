import argparse
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from reference import (
    FP4,
    FP8,
    assert_packed,
    cast,
    dequantize,
    quantize,
    read_packed,
    read_tensor,
    write_tensor,
)
from generate import generate

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/quantize"


def run(args, success=True):
    p = subprocess.run([str(BINARY), *map(str, args)], capture_output=True, text=True)
    assert (p.returncode == 0) == success, p.stdout + p.stderr
    return p


def main():
    global BINARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/correctness.json"
    )
    args = parser.parse_args()
    BINARY = args.binary.resolve()
    cases = []
    for fmt, dtype, out, tensor, stochastic in itertools.product(
        range(2), range(2), range(3), range(2), range(2)
    ):
        cases.append(
            (
                generate(7, 35, ["uniform", "normal", "outliers"][out], 42 + out),
                fmt,
                dtype,
                out,
                tensor,
                stochastic,
                32 if fmt == 0 else 16,
            )
        )
    if not args.quick:
        for fmt in range(2):
            for rows, cols in [
                (1, 1),
                (3, 15),
                (2, 16),
                (5, 31),
                (4, 32),
                (3, 64),
                (2, 129),
                (1, 1025),
            ]:
                for block in [16, 32, 64, 256, 1024]:
                    cases.append(
                        (generate(rows, cols, "normal"), fmt, 0, 0, False, False, block)
                    )
            for x in [
                np.zeros((3, 33), np.float32),
                np.array([[0.0, -0.0, 1e-35, -1e-35, 1e30, -1e30]], np.float32),
                np.concatenate((FP8, -FP8)).reshape(2, -1),
                np.concatenate((FP4, -FP4)).reshape(2, -1),
                np.concatenate(((FP8[1:] + FP8[:-1]) / 2, [448])).reshape(1, -1),
                np.concatenate(
                    (np.nextafter((FP8[1:] + FP8[:-1]) / 2, np.float32(np.inf)), [448])
                ).reshape(1, -1),
                np.concatenate(
                    (np.nextafter((FP8[1:] + FP8[:-1]) / 2, np.float32(-np.inf)), [448])
                ).reshape(1, -1),
            ]:
                cases.append((x, fmt, 0, 0, False, False, 32 if fmt == 0 else 16))
        # Tall tensors exercise grid.x row indexing beyond grid.y's 65535 limit;
        # isolated tiny/large blocks exercise E8M0 reciprocal edge cases.
        for fmt in range(2):
            for x in [
                generate(65537, 1, "normal"),
                np.full((2, 32), np.float32(1e-40)),
                np.full((2, 32), np.float32(1e38)),
            ]:
                cases.append((x, fmt, 0, 0, False, False, 32 if fmt == 0 else 16))
        # Hold global scale at 1 and local scale at each finite positive E4M3
        # value. Cover both signs of every FP4 midpoint and its +/-1 ULP values.
        boundaries = (FP4[1:] + FP4[:-1]) / 2
        scaled_cases = []
        for scale in FP8[1:]:
            midpoints = (boundaries * scale).astype(np.float32)
            points = np.concatenate(
                [
                    np.nextafter(midpoints, np.float32(-np.inf)),
                    midpoints,
                    np.nextafter(midpoints, np.float32(np.inf)),
                ]
            )
            points = np.concatenate([points, -points, np.zeros(3, np.float32)])
            blocks = np.empty((3, 16), np.float32)
            blocks[:, :15] = points.reshape(3, 15)
            blocks[:, 15] = scale * 6
            scaled_cases.append(blocks.reshape(-1))
        cases.append((np.stack(scaled_cases), 1, 0, 0, False, False, 16))
        # Tensor amax=448 fixes the MXFP8 scale at 1: exercise every raw code
        # and every RNE midpoint, without automatic block rescaling hiding it.
        mid = (FP8[1:] + FP8[:-1]) / 2
        for points in [
            FP8,
            mid,
            np.nextafter(mid, np.float32(np.inf)),
            np.nextafter(mid, np.float32(-np.inf)),
        ]:
            x = (
                np.concatenate((points, -points, [448, -448]))
                .astype(np.float32)
                .reshape(1, -1)
            )
            for stochastic in [False, True]:
                cases.append((x, 0, 0, 0, True, stochastic, 32))
        # Tensor amax=6 gives NVFP4 local scale 448 and global scale 1/448.
        points = np.concatenate((FP4, (FP4[1:] + FP4[:-1]) / 2))
        cases.append(
            (np.concatenate((points, -points)).reshape(1, -1), 1, 0, 0, True, False, 16)
        )
    if not args.quick:
        # Exercise both sides of C500's scalar/vector dispatch boundary, including
        # stochastic/tensor scaling and all dequantized output precisions.
        for rows, fmt, dtype, stochastic in itertools.product(
            [63, 64, 65], range(2), range(2), range(2)
        ):
            cases.append((generate(rows, 1024, "normal", rows), fmt, dtype,
                          rows % 3, rows == 64, stochastic, 32 if fmt == 0 else 16))
    with tempfile.TemporaryDirectory(prefix="lp-test-") as tmp:
        t = Path(tmp)
        for i, (x, fmt, dtype, out, tensor, stochastic, block) in enumerate(cases):
            write_tensor(t / "input", x, dtype)
            (t / "config").write_text(
                f'format = "{["mxfp8","nvfp4"][fmt]}"\nblock_size = {block}\nscale_mode = "{["block","tensor"][tensor]}"\noutput_type = "{["fp32","fp16","bf16"][out]}"\nrounding = "{["nearest","stochastic"][stochastic]}"\n'
            )
            run(
                [
                    "--input",
                    t / "input",
                    "--config",
                    t / "config",
                    "--packed",
                    t / "packed",
                    "--output",
                    t / "out",
                    "--log",
                    t / "log",
                    "--repeats",
                    2,
                ]
            )
            x, _ = read_tensor(t / "input")
            expected = quantize(x, fmt, block, tensor, stochastic)
            assert_packed(read_packed(t / "packed"), expected)
            actual, _ = read_tensor(t / "out")
            np.testing.assert_array_equal(actual, cast(dequantize(expected), out))
            metrics = json.loads((t / "log").read_text())
            err = x.astype(np.float64) - actual
            np.testing.assert_allclose(
                metrics["mae"], np.abs(err).mean(), rtol=1e-10, atol=1e-40
            )
            np.testing.assert_allclose(
                metrics["mse"], (err * err).mean(), rtol=1e-10, atol=1e-75
            )
            if i < 8:
                run(
                    [
                        "--mode",
                        "dequantize",
                        "--packed",
                        t / "packed",
                        "--output",
                        t / "reload",
                        "--output_type",
                        ["fp32", "fp16", "bf16"][out],
                        "--log",
                        t / "reload.json",
                        "--repeats",
                        2,
                    ]
                )
                assert (t / "reload").read_bytes() == (t / "out").read_bytes()
        # Public file boundary checks: truncated headers/payload and nonfinite input.
        (t / "bad").write_bytes(b"LPTENS1\0")
        base = [
            "--input",
            t / "bad",
            "--config",
            t / "config",
            "--packed",
            t / "p",
            "--output",
            t / "o",
            "--log",
            t / "l",
        ]
        run(base, False)
        write_tensor(t / "bad", np.array([[np.nan]], np.float32))
        run(base, False)
        (t / "bad").write_bytes((t / "packed").read_bytes()[:-1])
        run(
            [
                "--mode",
                "dequantize",
                "--packed",
                t / "bad",
                "--output",
                t / "o",
                "--log",
                t / "l",
            ],
            False,
        )
    summary = {
        "passed": len(cases),
        "negative_file_cases": 3,
        "oracle": "NumPy explicit FP8/FP4 codebook search",
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
