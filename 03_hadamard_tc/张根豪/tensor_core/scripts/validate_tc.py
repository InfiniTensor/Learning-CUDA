#!/usr/bin/env python3
"""Full-tensor PDF checks. Never substitute relative error for absolute error.

Default reference is Dao-AILab fast_hadamard_transform. --reference cpu is an
explicit independent FP32 butterfly reference, recorded separately, not a
claim that the external library ran. Quant fusion checks are emitted by C++.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


def cases(smoke=False):
    dims = (64, 128, 256) if smoke else (64, 128, 256, 512, 1024)
    sizes = (1, 5) if smoke else (1, 5, 2048)
    for dtype in ("fp16", "bf16"):
        for dim in dims:
            for normalize in (True, False):
                for tokens in sizes:
                    yield dtype, dim, normalize, tokens, "normal"
    if not smoke:
        for dtype in ("fp16", "bf16"):
            for dim in (64, 128, 256):
                for normalize in (True, False):
                    for pattern in ("zeros", "constant", "alternating", "impulse", "uniform", "outlier"):
                        yield dtype, dim, normalize, 5, pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", choices=("library", "cpu"), default="library")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import torch
    library = None
    if args.reference == "library":
        from fast_hadamard_transform import hadamard_transform
        library = hadamard_transform
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    all_pass = True
    infrastructure_ok = True
    torch.set_num_threads(2)
    metadata = {"reference": args.reference, "torch": torch.__version__,
                "cuda_build": torch.version.cuda, "seed": 20260908,
                "thresholds": {"fp16": 0.01, "bf16": 0.05}, "full_tensor": True}
    (args.output_dir / "validation_environment.json").write_text(json.dumps(metadata, indent=2))
    for index, (dtype, dim, norm, tokens, pattern) in enumerate(cases(args.smoke)):
        torch.manual_seed(20260908 + index)
        native = torch.float16 if dtype == "fp16" else torch.bfloat16
        x = torch.randn(tokens, dim)
        if pattern == "zeros": x.zero_()
        elif pattern == "constant": x.fill_(0.25)
        elif pattern == "alternating": x[:] = (torch.arange(dim) % 2 * 2 - 1) * 0.25
        elif pattern == "impulse":
            x.zero_()
            x[:, 0] = 1
        elif pattern == "uniform": x.uniform_(-1, 1)
        elif pattern == "outlier":
            x.mul_(0.01)
            x[:, 0] = 32  # finite, no output overflow; exercises dynamic range
        x = x.to(native).contiguous()
        scale = 1 / math.sqrt(dim) if norm else 1
        if library:
            ref = library(x.cuda(), scale).cpu().contiguous()
        else:
            ref = x.float().clone()
            stride = 1
            while stride < dim:
                blocks = ref.reshape(tokens, -1, 2, stride)
                a, b = blocks[:, :, 0].clone(), blocks[:, :, 1].clone()
                blocks[:, :, 0], blocks[:, :, 1] = a + b, a - b
                stride *= 2
            ref = (ref * scale).to(native).contiguous()
        name = f"{index:03d}_{dtype}_d{dim}_n{tokens}_norm{int(norm)}_{pattern}"
        with tempfile.TemporaryDirectory(prefix="tc-case-", dir=args.output_dir) as tmp:
            tmp = Path(tmp)
            (tmp / "input.bin").write_bytes(x.view(torch.uint16).numpy().tobytes())
            (tmp / "ref.bin").write_bytes(ref.view(torch.uint16).numpy().tobytes())
            command = [str(args.bin.resolve()), "--batch", "1", "--seq", "1", "--heads", str(tokens),
                       "--head_dim", str(dim), "--dtype", dtype, "--normalize", str(norm).lower(),
                       "--warmup", "0", "--iters", "1", "--ref_tokens", str(tokens),
                       "--input_bin", str(tmp / "input.bin"), "--reference_bin", str(tmp / "ref.bin"),
                       "--dump_dir", str(tmp / "dump"), "--csv", str(tmp / "rows.csv")]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (args.output_dir / f"{name}.log").write_text(process.stdout + process.stderr)
            if not (tmp / "rows.csv").exists():
                infrastructure_ok = False
                all_pass = False
                print(name, "EXECUTION_ERROR", process.returncode, flush=True)
                continue
            with (tmp / "rows.csv").open() as source:
                rows = list(csv.DictReader(source))
            if index == 0:
                # Negative I/O/acceptance tests. A forged reference must fail
                # even if the separate FP64 diagnostic would pass.
                negative = []
                for label, extra, expected_rc in (
                    ('zero_ref_tokens', ['--ref_tokens', '0'], 2),
                    ('negative_batch', ['--batch', '-1'], 2),
                    ('overflow_shape', ['--batch', '2147483647', '--seq', '2147483647'], 2),
                ):
                    result = subprocess.run(command + extra, capture_output=True, text=True, timeout=120)
                    negative.append({'test': label, 'pass': result.returncode == expected_rc})
                input_bytes = (tmp / 'input.bin').read_bytes()
                for label, bad_bytes in (('short_input', input_bytes[:-1]), ('long_input', input_bytes + b'\x00')):
                    (tmp / 'bad.bin').write_bytes(bad_bytes)
                    result = subprocess.run(command + ['--input_bin', str(tmp / 'bad.bin')], capture_output=True, text=True, timeout=120)
                    negative.append({'test': label, 'pass': result.returncode == 2})
                (tmp / 'forged.bin').write_bytes(bytes(len(input_bytes)))
                result = subprocess.run(command + ['--reference_bin', str(tmp / 'forged.bin'),
                                                  '--dump_dir', str(tmp / 'forged_dump'),
                                                  '--csv', str(tmp / 'forged.csv')],
                                        capture_output=True, text=True, timeout=120)
                forged_rows = []
                if (tmp / 'forged.csv').exists():
                    with (tmp / 'forged.csv').open() as source:
                        forged_rows = [r for r in csv.DictReader(source) if r['kind'] == 'transform']
                negative.append({'test': 'forged_reference', 'pass': result.returncode == 1 and
                                 len(forged_rows) == 5 and all(r['pdf_verified'] == '0' for r in forged_rows)})
                infrastructure_ok &= all(r['pass'] for r in negative)
                (args.output_dir / 'negative_tests.json').write_text(json.dumps(negative, indent=2))
            case_pass = True
            for row in rows:
                impl = row["implementation"]
                record = {"case": name, "dtype": dtype, "head_dim": dim, "tokens": tokens,
                          "normalize": norm, "pattern": pattern, "reference": args.reference,
                          "implementation": impl, "kind": row["kind"], "max_abs": "",
                          "pass": "", "sha256": "", "harness_consistent": True}
                if row["kind"] == "transform":
                    raw = (tmp / "dump" / f"{impl}.bin").read_bytes()
                    actual = torch.frombuffer(bytearray(raw), dtype=native).reshape(tokens, dim).float()
                    error = (actual - ref.float()).abs().max().item()
                    passed = math.isfinite(error) and error < (0.01 if dtype == "fp16" else 0.05)
                    record.update(max_abs=error, **{"pass": passed}, sha256=hashlib.sha256(raw).hexdigest())
                    consistent = (row["pdf_verified"] == "1") == passed
                    consistent &= math.isclose(float(row["reference_max_abs_error"]), error, rel_tol=1e-5, abs_tol=1e-12)
                    consistent &= int(row["validated_elements"]) == tokens * dim
                    record["harness_consistent"] = consistent
                    infrastructure_ok &= consistent
                else:
                    passed = row["quant_vs_unfused"] in ("reference", "bit-exact")
                    record["pass"] = passed
                case_pass &= passed
                records.append(record)
            infrastructure_ok &= process.returncode == (0 if case_pass else 1)
            all_pass &= case_pass
            print(name, "PASS" if case_pass else "FAIL(abs or fusion)", flush=True)
            # Preserve failing inputs to allow direct replay without the generator.
            if not case_pass:
                failure = args.output_dir / "failures" / name
                failure.mkdir(parents=True, exist_ok=True)
                for src in ("input.bin", "ref.bin"):
                    (failure / src).write_bytes((tmp / src).read_bytes())
    if records:
        with (args.output_dir / "validation.csv").open("w", newline="") as dest:
            writer = csv.DictWriter(dest, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {"cases": len(set(r["case"] for r in records)), "rows": len(records),
               "failed_rows": sum(not r["pass"] for r in records),
               "harness_consistent": infrastructure_ok, "all_pass": all_pass,
               "reference": args.reference}
    (args.output_dir / "validation_status.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0 if all_pass and infrastructure_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
