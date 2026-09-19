#!/usr/bin/env python3
"""Compare against the real, commit-pinned Dao-AILab CUDA implementation.

No local oracle is substituted. Missing dependencies, GPU, or provenance return
nonzero and write an ERROR report. Numerical failures write raw metrics and
reproducible CPU tensor witnesses before returning nonzero.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.parse

from build_torch_extension import load_extension

REFERENCE_COMMIT = "e7706faf8d1c3b9f241e36860640ad1dac644ede"
REFERENCE_REPO = "https://github.com/Dao-AILab/fast-hadamard-transform"


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(package, backend, repository):
    direct = {}
    try:
        distribution = importlib.metadata.distribution("fast_hadamard_transform")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        pass
    vcs = direct.get("vcs_info", {})
    url = urllib.parse.urlsplit(direct.get("url", ""))
    official_url = url.hostname == "github.com" and url.path.rstrip("/").removesuffix(".git").lower() == "/dao-ailab/fast-hadamard-transform"
    verified = official_url and vcs.get("commit_id") == REFERENCE_COMMIT
    evidence = "PEP610 VCS installation" if verified else None
    if repository is not None:
        repository = Path(repository).resolve()
        def git(*arguments):
            return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True, timeout=20).strip()
        commit = git("rev-parse", "HEAD")
        if commit != REFERENCE_COMMIT:
            raise RuntimeError(f"reference checkout is {commit}, expected {REFERENCE_COMMIT}")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("reference checkout has modified tracked files")
        module_local = Path(package.__file__).resolve().is_relative_to(repository)
        backend_local = Path(backend.__file__).resolve().is_relative_to(repository)
        direct_local = url.scheme == "file" and Path(urllib.parse.unquote(url.path)).resolve() == repository
        if (module_local and backend_local) or direct_local:
            verified = True
            evidence = "clean pinned source checkout and local module/install provenance"
    if not verified:
        raise RuntimeError("Cannot verify the reference commit. Install the pinned VCS URL with "
                           "FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE, or pass --reference-repo "
                           "for a clean pinned checkout used for an in-place/local build.")
    return {"repository": REFERENCE_REPO, "commit": REFERENCE_COMMIT,
            "verification": evidence, "package_version": getattr(package, "__version__", None),
            "python_module": str(Path(package.__file__).resolve()),
            "cuda_module": str(Path(backend.__file__).resolve()),
            "cuda_module_sha256": hash_file(backend.__file__)}


def make_input(torch, shape, dtype, pattern, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if pattern == "uniform":
        values = torch.rand(shape, generator=generator, dtype=torch.float32) * 2 - 1
    elif pattern == "normal":
        values = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5
    elif pattern == "outlier":
        values = (torch.rand(shape, generator=generator, dtype=torch.float32) * 2 - 1) * 0.001
        rows = values.reshape(-1, shape[-1])
        rows[torch.arange(rows.shape[0]), torch.arange(rows.shape[0]) * 7 % shape[-1]] = 8
    elif pattern == "zeros":
        values = torch.zeros(shape, dtype=torch.float32)
    else:
        raise ValueError(pattern)
    return values.to(device=device, dtype=dtype).contiguous()


def metrics(torch, actual, expected, tolerance):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise RuntimeError("reference/ours output shape, dtype, or device mismatch")
    actual_cpu, expected_cpu = actual.float().cpu(), expected.float().cpu()
    errors = (actual_cpu - expected_cpu).abs()
    finite = bool(torch.isfinite(errors).all())
    worst = int(torch.nan_to_num(errors, nan=float("inf")).reshape(-1).argmax())
    maximum = float(errors.reshape(-1)[worst]) if finite else None
    failed = int((~torch.isfinite(errors) | (errors >= tolerance)).sum())
    return {"pass": finite and failed == 0, "max_abs_error": maximum,
            "mean_abs_error": float(errors.mean()) if finite else None,
            "mismatching_elements": int((actual_cpu != expected_cpu).sum()),
            "elements_at_or_above_strict_limit": failed, "strict_abs_limit": tolerance,
            "worst_flat_index": worst,
            "ours_at_worst": float(actual_cpu.reshape(-1)[worst]) if finite else None,
            "reference_at_worst": float(expected_cpu.reshape(-1)[worst]) if finite else None}


def check_rejections(torch, extension, device):
    cases = {
        "cpu": lambda: extension.hadamard(torch.ones((2, 16), dtype=torch.float16)),
        "dtype": lambda: extension.hadamard(torch.ones((2, 16), device=device, dtype=torch.float32)),
        "rank": lambda: extension.hadamard(torch.ones((1, 2, 16), device=device, dtype=torch.float16)),
        "noncontiguous": lambda: extension.hadamard(torch.ones((16, 16), device=device, dtype=torch.float16).t()),
        "non_power_of_two": lambda: extension.hadamard(torch.ones((2, 3), device=device, dtype=torch.float16)),
        "dimension_too_large": lambda: extension.hadamard(torch.ones((2, 512), device=device, dtype=torch.float16)),
        "empty": lambda: extension.hadamard(torch.empty((0, 16), device=device, dtype=torch.float16)),
        "scale_nan": lambda: extension.hadamard(torch.ones((2, 16), device=device, dtype=torch.float16), float("nan")),
        "scale_zero": lambda: extension.hadamard(torch.ones((2, 16), device=device, dtype=torch.float16), 0),
        "autograd": lambda: extension.hadamard(torch.ones((2, 16), device=device, dtype=torch.float16, requires_grad=True)),
    }
    results = []
    for name, operation in cases.items():
        try:
            operation()
        except (RuntimeError, ValueError) as error:
            results.append({"case": name, "pass": True, "error": str(error).splitlines()[0]})
        else:
            raise RuntimeError(f"extension accepted invalid input: {name}")
    return results


def benchmark_pair(torch, ours, reference, values, scale, groups, repetitions, warmup):
    functions = {"ours": lambda: ours(values, scale), "dao": lambda: reference(values, scale)}
    for function in functions.values():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize(values.device)
    samples = {name: [] for name in functions}
    for group in range(groups):
        order = ("ours", "dao") if group % 2 == 0 else ("dao", "ours")
        for name in order:
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repetitions):
                functions[name]()
            finish.record()
            finish.synchronize()
            samples[name].append(begin.elapsed_time(finish) * 1000 / repetitions)
    medians = {name: statistics.median(times) for name, times in samples.items()}
    return {"method": "eager_api", "samples_us": samples, "median_us": medians,
            "dao_over_ours": medians["dao"] / medians["ours"],
            "groups": groups, "repetitions_per_group": repetitions, "warmup": warmup,
            "scope": "CUDA-event interval around allocating PyTorch API calls, including GPU idle gaps from host dispatch; no H2D, D2H, build, or validation"}


def benchmark_graph_pair(torch, ours, reference, values, scale, groups, captured_calls,
                         replays, warmup, graph_warmup):
    """Measure captured GPU work with Python dispatch amortized over many calls."""
    functions = {"ours": lambda: ours(values, scale), "dao": lambda: reference(values, scale)}
    graphs, outputs, expected = {}, {}, {}
    capture_stream = torch.cuda.Stream(device=values.device)
    capture_stream.wait_stream(torch.cuda.current_stream(values.device))
    with torch.cuda.stream(capture_stream):
        for name, function in functions.items():
            for _ in range(warmup):
                function()
            expected[name] = function()
    capture_stream.synchronize()
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        # Retain every output for both implementations. Each captured call gets
        # its own allocation, and replays reuse those fixed allocations. Separate
        # private graph pools allow the alternating measurement order below.
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs[name] = [function() for _ in range(captured_calls)]
        graphs[name] = graph
    torch.cuda.synchronize(values.device)
    for graph in graphs.values():
        # Always execute once before validating, even when warmup is disabled.
        graph.replay()
    torch.cuda.synchronize(values.device)
    for name, tensors in outputs.items():
        for index, output in enumerate(tensors):
            if not torch.equal(output, expected[name]):
                raise RuntimeError(f"CUDA graph output differs from eager output: {name}, call {index}")

    for graph in graphs.values():
        for _ in range(graph_warmup):
            graph.replay()
    torch.cuda.synchronize(values.device)

    intervals = {name: [] for name in functions}
    samples = {name: [] for name in functions}
    orders = []
    for group in range(groups):
        order = ("ours", "dao") if group % 2 == 0 else ("dao", "ours")
        orders.append(list(order))
        for name in order:
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(replays):
                graphs[name].replay()
            finish.record()
            finish.synchronize()
            elapsed_ms = begin.elapsed_time(finish)
            intervals[name].append(elapsed_ms)
            samples[name].append(elapsed_ms * 1000 / (replays * captured_calls))
    medians = {name: statistics.median(times) for name, times in samples.items()}
    return {"method": "cuda_graph", "samples_us": samples, "median_us": medians,
            "raw_event_intervals_ms": intervals,
            "dao_over_ours": medians["dao"] / medians["ours"],
            "groups": groups, "captured_calls_per_graph": captured_calls,
            "replays_per_group": replays, "calls_per_group": replays * captured_calls,
            "api_warmup_calls": warmup, "graph_warmup_replays": graph_warmup,
            "validation_replays": 1, "all_captured_outputs_equal_eager": True,
            "group_order": orders,
            "output_lifetime": "One output per captured call retained through measurement; separate private graph pools; fixed input and output addresses on replay",
            "scope": "CUDA-event interval around graph replays divided by captured calls; includes captured GPU work and amortized replay scheduling, excludes per-call Python dispatch and allocation, capture, validation, H2D and D2H; not standalone kernel latency"}


def run(args, report):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-enabled PyTorch/GPU is available; reference comparison was not run")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device")
    torch.cuda.set_device(device)
    # The checkout verifies provenance; imports must resolve to the installed
    # package. Prepending the source tree can shadow its PEP 610 metadata with
    # setup.py's source egg-info and incorrectly reject a verified local build.
    import fast_hadamard_transform as reference_package
    import fast_hadamard_transform_cuda as reference_backend
    reference = reference_package.hadamard_transform
    report["reference"] = provenance(reference_package, reference_backend, args.reference_repo)
    extension = load_extension(args.verbose, args.build_directory)
    report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "torch_cuda": torch.version.cuda, "device": str(device),
                             "gpu": torch.cuda.get_device_name(device),
                             "compute_capability": list(torch.cuda.get_device_capability(device)),
                             "extension": str(Path(extension.__file__).resolve()),
                             "extension_sha256": hash_file(extension.__file__)}
    report["rejected_inputs"] = check_rejections(torch, extension, device)
    report["cases"] = []
    failures = 0
    with torch.inference_mode():
        for dtype_name, dtype, limit in (("fp16", torch.float16, 1e-2), ("bf16", torch.bfloat16, 5e-2)):
            for dim in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                for shape in ((1, dim), (3, dim), (17, dim), (1, 3, 7, dim), (2, 5, 13, dim)):
                    for pattern in ("uniform", "normal", "outlier", "zeros"):
                        seeds = (2026,) if pattern == "zeros" else (2026, 95811, 314159)
                        for seed in seeds:
                            values = make_input(torch, shape, dtype, pattern, seed, device)
                            for normalized in (False, True):
                                scale = float(torch.tensor(1 / math.sqrt(dim) if normalized else 1, dtype=torch.float32))
                                actual = extension.hadamard(values, scale)
                                expected = reference(values, scale)
                                result = metrics(torch, actual, expected, limit)
                                packed, scales = extension.hadamard_int4(values, scale)
                                split_packed, split_scales = extension.quantize_int4(actual)
                                quant_exact = bool(torch.equal(packed, split_packed) and torch.equal(scales, split_scales))
                                result.update({"dtype": dtype_name, "shape": list(shape), "pattern": pattern,
                                               "seed": seed, "normalized": normalized, "scale": scale,
                                               "fused_vs_split_int4_exact": quant_exact})
                                result["pass"] = result["pass"] and quant_exact
                                if not result["pass"]:
                                    failures += 1
                                    witness_directory = Path(args.json).with_suffix("").with_name(Path(args.json).stem + "_failures")
                                    witness_directory.mkdir(parents=True, exist_ok=True)
                                    witness = witness_directory / f"case_{len(report['cases']):05d}.pt"
                                    torch.save({"input": values.cpu(), "ours": actual.cpu(), "dao": expected.cpu(),
                                                "absolute_error": (actual.float() - expected.float()).abs().cpu(),
                                                "metadata": result}, witness)
                                    result["witness"] = str(witness)
                                report["cases"].append(result)
                print(f"CHECKED dtype={dtype_name} dim={dim} cases={len(report['cases'])} failures={failures}", flush=True)
        # Data production and the wrapper execute on a non-default stream; the
        # reference is evaluated only after this stream completes.
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            stream_input = torch.arange(17 * 256, device=device, dtype=torch.float32).remainder(29).div(32).to(torch.float16).reshape(17, 256)
            stream_output = extension.hadamard(stream_input, 1.0)
        stream.synchronize()
        report["non_default_stream"] = metrics(torch, stream_output, reference(stream_input, 1.0), 1e-2)
        if not report["non_default_stream"]["pass"]:
            failures += 1
        report["multi_device_guard"] = "not_exercised: requires a second visible GPU"
        if torch.cuda.device_count() > 1:
            original = torch.cuda.current_device()
            other = (original + 1) % torch.cuda.device_count()
            values = torch.ones((3, 16), device=f"cuda:{other}", dtype=torch.float16)
            guarded_output = extension.hadamard(values)
            if guarded_output.device.index != other or torch.cuda.current_device() != original:
                raise RuntimeError("device guard failed to preserve caller device/output device")
            report["multi_device_guard"] = "PASS"
        report["benchmarks"] = []
        report["graph_benchmarks"] = []
        if args.benchmark:
            for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
                for dim in (16, 64, 256):
                    for shape in ((17, dim), (4, 128, 8, dim)):
                        values = make_input(torch, shape, dtype, "normal", 2026, device)
                        entry = benchmark_pair(torch, extension.hadamard, reference, values, 1.0,
                                               args.groups, args.repetitions, args.warmup)
                        entry.update({"dtype": dtype_name, "shape": list(shape), "scale": 1.0})
                        report["benchmarks"].append(entry)
                        graph_entry = benchmark_graph_pair(
                            torch, extension.hadamard, reference, values, 1.0,
                            args.groups, args.graph_calls, args.graph_replays,
                            args.warmup, args.graph_warmup)
                        graph_entry.update({"dtype": dtype_name, "shape": list(shape), "scale": 1.0})
                        report["graph_benchmarks"].append(graph_entry)
    maxima = {}
    for dtype in ("fp16", "bf16"):
        errors = [case["max_abs_error"] for case in report["cases"] if case["dtype"] == dtype]
        maxima[dtype] = None if any(error is None for error in errors) else max(errors)
    report["summary"] = {"cases": len(report["cases"]), "failures": failures,
                         "max_abs_error_by_dtype": maxima}
    report["status"] = "PASS" if failures == 0 else "FAIL"
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="results/third_party_reference.json")
    parser.add_argument("--reference-repo", help="Optional clean pinned checkout used to build the installed reference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", action="store_true", help="Run both eager API and CUDA Graph benchmarks")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--graph-calls", type=int, default=64, help="Independent API calls captured in each CUDA graph")
    parser.add_argument("--graph-replays", type=int, default=20, help="CUDA graph replays per timing group")
    parser.add_argument("--graph-warmup", type=int, default=5, help="Untimed graph warmup replays after validating one replay")
    parser.add_argument("--build-directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if min(args.groups, args.repetitions, args.graph_calls, args.graph_replays) < 1 or min(args.warmup, args.graph_warmup) < 0:
        parser.error("groups/repetitions/graph-calls/graph-replays must be positive and warmups nonnegative")
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "reference_commit_required": REFERENCE_COMMIT,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    start = time.monotonic()
    try:
        code = run(args, report)
    except Exception as error:
        report.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}"})
        code = 2
    report["elapsed_seconds"] = time.monotonic() - start
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "json": str(output),
                      "summary": report.get("summary"), "error": report.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
