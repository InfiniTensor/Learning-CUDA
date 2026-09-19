#!/usr/bin/env python3
"""Check stream ordering and changing-input CUDA Graph replay against a CPU oracle.

Negative controls deliberately use the default stream or retain a stale graph
output. A run passes only if both faults are detected for every configuration.
This is correctness coverage on one device, not a performance measurement.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from build_torch_extension import load_extension

DELAY_CYCLES = 50_000_000
SHAPES = ((3, 1), (17, 8), (17, 64), (17, 256), (2, 1, 3, 256))


def configurations():
    for shape in SHAPES:
        for threads in (128, 256):
            common = {"shape": shape, "block_threads": threads}
            yield dict(common, method="quantize_int4")
            layouts = ("original", "auto", "packed") if shape[-1] <= 16 else ("original", "auto")
            for method in ("hadamard", "hadamard_int4"):
                for layout in layouts:
                    yield dict(common, method=method, row_layout=layout)
            if shape[-1] == 256 and threads == 128:
                yield dict(common, method="hadamard_int4", fused_layout="contiguous256")


def invoke(op, x, case):
    keywords = {key: value for key, value in case.items() if key not in ("shape", "method")}
    if case["method"] != "quantize_int4":
        keywords["scale"] = 1 / math.sqrt(case["shape"][-1])
    return getattr(op, case["method"])(x, **keywords)


def fixture(torch, shape, dtype, phase):
    index = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    if phase == 0:
        values = index * 0
    elif phase == 1:
        values = (index.remainder(29) - 14) / 16
    elif phase == 2:
        values = (index.remainder(17) - 8) / 8 + 1 / 16
    else:
        values = (index.remainder(11) - 5) / 4 - 1 / 8
    return values.to(dtype)


def reference(torch, np, cpu_input, case):
    values = cpu_input.float().numpy().copy()
    n = values.shape[-1]
    if case["method"] != "quantize_int4":
        stride = 1
        while stride < n:
            blocks = values.reshape(-1, n // (2 * stride), 2 * stride)
            left, right = blocks[..., :stride].copy(), blocks[..., stride:].copy()
            blocks[..., :stride], blocks[..., stride:] = left + right, left - right
            stride *= 2
        # The fused contract quantizes the rounded FP16/BF16 transform output.
        rounded = torch.from_numpy(values * np.float32(1 / math.sqrt(n))).to(cpu_input.dtype)
        if case["method"] == "hadamard":
            return (rounded,)
        values = rounded.float().numpy()
    maximum = np.max(np.abs(values), axis=-1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(7)).astype(np.float32)
    quantized = np.clip(np.rint(values / scales[..., None]), -7, 7).astype(np.int32)
    padded = np.zeros((*quantized.shape[:-1], (n + 1) // 2 * 2), dtype=np.int32)
    padded[..., :n] = quantized
    packed = ((padded[..., ::2] & 15) | ((padded[..., 1::2] & 15) << 4)).astype(np.uint8)
    return torch.from_numpy(packed), torch.from_numpy(scales)


def outputs(value):
    return value if isinstance(value, tuple) else (value,)


def compare(torch, actual, expected):
    actual = outputs(actual)
    if len(actual) != len(expected):
        raise AssertionError("output tuple length differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        left = left.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"output {index} shape/dtype differs")
        if not torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)):
            raise AssertionError(f"output {index} bits differ from the CPU oracle")


def require_rejection(torch, bad_output, expected, fault):
    try:
        compare(torch, bad_output, expected)
    except AssertionError as error:
        return {"fault": fault, "detected": True, "reason": str(error)}
    raise AssertionError(f"negative control was not detected: {fault}")


def stream_probe(torch, op, case, cpu_input, wrong_stream):
    x, payload = torch.zeros_like(cpu_input, device="cuda"), cpu_input.cuda()
    side, default = torch.cuda.Stream(), torch.cuda.default_stream()
    # Warm allocations on both streams before the delayed producer. All tensors
    # remain alive until synchronization, so allocator reuse cannot race a free.
    for stream in (default, side):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = invoke(op, x, case)
                del warm
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(DELAY_CYCLES)
        x.copy_(payload, non_blocking=True)
        if wrong_stream:
            # Test-only fault: remove the producer/consumer stream ordering.
            with torch.cuda.stream(default):
                result = invoke(op, x, case)
        else:
            result = invoke(op, x, case)
    torch.cuda.synchronize()
    compare(torch, x, (cpu_input,))
    return result


def check_case(torch, np, op, case, dtype):
    cpu_inputs = [fixture(torch, case["shape"], dtype, phase) for phase in range(4)]
    expected = [reference(torch, np, value, case) for value in cpu_inputs]
    good = stream_probe(torch, op, case, cpu_inputs[1], False)
    compare(torch, good, expected[1])
    bad = stream_probe(torch, op, case, cpu_inputs[1], True)
    stream_control = require_rejection(torch, bad, expected[1], "consumer_on_default_stream")

    x = cpu_inputs[0].cuda()
    payloads = [value.cuda() for value in cpu_inputs]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            warm = invoke(op, x, case)
            del warm
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        result = invoke(op, x, case)
    stale = None
    graph_control = None
    replays = []
    for phase, payload in enumerate(payloads):
        with torch.cuda.stream(side):
            x.copy_(payload, non_blocking=True)
            graph.replay()
        side.synchronize()
        compare(torch, result, expected[phase])
        compare(torch, x, (cpu_inputs[phase],))
        if phase == 0:
            stale = tuple(value.cpu().clone() for value in outputs(result))
        elif phase == 1:
            graph_control = require_rejection(torch, stale, expected[phase], "stale_graph_output")
        replays.append({"phase": phase, "oracle_bitwise_equal": True, "input_unchanged": True})
    return dict(case, dtype=str(dtype), status="PASS", stream_oracle_bitwise_equal=True,
                stream_input_unchanged=True, graph_replays=replays,
                negative_controls=[stream_control, graph_control])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        stream = args.json.open("x", encoding="utf-8")
    except OSError as error:
        parser.error(f"cannot create output: {error}; choose a new path")
    report = {"status": "RUNNING", "cases": [], "delay_cycles": DELAY_CYCLES,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "limitations": ["Single-device correctness only; no speedup claim.",
                              "Delay uses the test-only torch.cuda._sleep API; unavailable or undetected controls fail.",
                              "Representative shapes/layouts, not every possible stream schedule."]}

    def save():
        stream.seek(0)
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stream.truncate()
        stream.flush()

    with stream:
        save()
        try:
            import numpy as np
            import torch
            if not torch.cuda.is_available() or torch.version.hip:
                raise RuntimeError("a CUDA-enabled NVIDIA GPU is required")
            if not callable(getattr(torch.cuda, "_sleep", None)):
                raise RuntimeError("torch.cuda._sleep is required for the negative control")
            op = load_extension(verbose=True, build_directory=str(args.build_directory))
            report["environment"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                     "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
                                     "capability": list(torch.cuda.get_device_capability()),
                                     "extension_sha256": hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest()}
            root = Path(__file__).resolve().parents[1]
            report["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file()
                and path.suffix in (".cu", ".cuh") and not {"build", "results"}.intersection(path.relative_to(root).parts)}
            for name in ("scripts/build_torch_extension.py", "scripts/verify_execution_context.py"):
                report["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            for dtype in (torch.float16, torch.bfloat16):
                for case in configurations():
                    report["active_case"] = dict(case, dtype=str(dtype))
                    save()
                    report["cases"].append(check_case(torch, np, op, case, dtype))
            report.pop("active_case", None)
            report["status"] = "PASS"
        except KeyboardInterrupt:
            report.update(status="INTERRUPTED", traceback=traceback.format_exc())
        except Exception:
            report.update(status="FAIL", traceback=traceback.format_exc())
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    print(json.dumps({"status": report["status"], "configurations": len(report["cases"])}))
    if report["status"] != "PASS":
        print(report["traceback"])
        return 130 if report["status"] == "INTERRUPTED" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
