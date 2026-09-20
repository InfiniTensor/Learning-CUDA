import argparse
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from reference import (
    assert_packed,
    cast,
    hadamard,
    quantize,
    read_packed,
    read_tensor,
    write_tensor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build/hadamard")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/correctness.json"
    )
    args = parser.parse_args()
    cases = list(
        itertools.product(
            [1, 2], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [0, 1], [19]
        )
    )
    # Few-row/CTA-boundary cases exercise collective MMA and block reductions
    # with inactive warps. Keep these extra cases normalized.
    cases += list(itertools.product([1, 2], [64, 1024], [0, 1], [1, 2, 31, 32]))
    # Larger aligned shapes and tails exercise many-CTA reductions and
    # partially populated final matrix waves, for both device backends.
    for dtype, fmt in itertools.product([1, 2], [0, 1]):
        cases += [(dtype, 1024, fmt, rows) for rows in [2047, 2048, 2049]]
        cases.append((dtype, 64, fmt, 8193))
    maxerr = {1: 0.0, 2: 0.0}
    mma_maxerr = {1: 0.0, 2: 0.0}
    mma_cases = 0
    with tempfile.TemporaryDirectory(prefix="had-test-") as tmp:
        t = Path(tmp)
        for i, (dtype, d, fmt, rows) in enumerate(cases):
            rng = np.random.default_rng(i)
            x = rng.normal(size=(rows, d)).astype(np.float32)
            if rows > 1:
                x[0] = 0
                x[1] = 0
                x[1, 0] = 8
            signs = i % 2 == 0
            stochastic = i % 3 == 0
            norm = i >= 44 or i % 4 != 0
            write_tensor(t / "input", x, dtype)
            x, _ = read_tensor(t / "input")
            command = [
                str(args.binary.resolve()),
                "--input",
                str(t / "input"),
                "--output",
                str(t / "output"),
                "--packed",
                str(t / "packed"),
                "--unfused_packed",
                str(t / "unfused"),
                "--log",
                str(t / "log"),
                "--format",
                ["mxfp8", "nvfp4"][fmt],
                "--rounding",
                "stochastic" if stochastic else "nearest",
                "--normalize",
                str(int(norm)),
                "--random_sign",
                str(int(signs)),
                "--repeats",
                "3",
                "--tc_output",
                str(t / "tc"),
                "--factorized_tc_output",
                str(t / "mma"),
                "--factorized_tc_packed",
                str(t / "mma_packed"),
            ]
            command += [
                "--materialized_compare",
                "1",
                "--factorized_tc_materialized_packed",
                str(t / "materialized"),
            ]
            (t / "mma").unlink(missing_ok=True)
            p = subprocess.run(command, capture_output=True, text=True)
            assert p.returncode == 0, (dtype, d, fmt, p.stdout, p.stderr)
            y, _ = read_tensor(t / "output")
            ref = cast(hadamard(x, norm, signs), dtype)
            e = float(np.abs(y - ref).max())
            maxerr[dtype] = max(maxerr[dtype], e)
            assert e < (1e-2 if dtype == 1 else 5e-2), (dtype, d, e)
            expected = quantize(y, fmt, 32 if fmt == 0 else 16, stochastic=stochastic)
            assert_packed(read_packed(t / "packed"), expected)
            assert (t / "packed").read_bytes() == (t / "unfused").read_bytes()
            metrics = json.loads((t / "log").read_text())
            if dtype == 1 and d >= 16 and metrics["platform"] != "ascend_cann":
                assert "tensor_core_ms" in metrics
            if d >= 16 and metrics["platform"] == "ascend_cann":
                assert "factorized_tc_ms" in metrics
            if "tensor_core_ms" in metrics:
                tc, _ = read_tensor(t / "tc")
                assert np.max(np.abs(tc - ref)) < 1e-2
            if "factorized_tc_ms" in metrics:
                mma, _ = read_tensor(t / "mma")
                error = float(np.max(np.abs(mma - ref)))
                assert error < (1e-2 if dtype == 1 else 5e-2), (dtype, d, error)
                mma_maxerr[dtype] = max(mma_maxerr[dtype], error)
                mma_expected = quantize(
                    mma, fmt, 32 if fmt == 0 else 16, stochastic=stochastic
                )
                assert_packed(read_packed(t / "mma_packed"), mma_expected)
                if fmt == 1:
                    assert_packed(read_packed(t / "materialized"), mma_expected)
                mma_cases += 1
        write_tensor(t / "input", np.ones((2, 63)), 1)
        p = subprocess.run(command, capture_output=True, text=True)
        assert p.returncode != 0 and "power of two" in p.stderr
    result = {
        "status": "PASS",
        "passed": len(cases),
        "invalid_shape_cases": 1,
        "max_abs_error_fp16": maxerr[1],
        "max_abs_error_bf16": maxerr[2],
        "fused_packed_equal": True,
        "factorized_tc_cases": mma_cases,
        "factorized_tc_max_abs_error_fp16": mma_maxerr[1],
        "factorized_tc_max_abs_error_bf16": mma_maxerr[2],
        "oracle": "NumPy float64 dense Sylvester matrix",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
