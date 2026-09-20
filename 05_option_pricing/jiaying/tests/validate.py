#!/usr/bin/env python3
"""Independent analytical checks plus CPU/GPU/RNG/launch-layout regression tests."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def config(path, values):
    path.write_text("\n".join(f"{k} = {v}" for k, v in values.items()) + "\n")


def bs(spot=100, strike=100, rate=.03, sigma=.2, maturity=1, put=False):
    sign = -1 if put else 1
    if not maturity or not sigma:
        return max(sign * (spot - strike * math.exp(-rate * maturity)), 0)
    d1 = (math.log(spot / strike) + (rate + sigma * sigma / 2) * maturity) / (sigma * math.sqrt(maturity))
    d2 = d1 - sigma * math.sqrt(maturity)
    cdf = lambda x: .5 * math.erfc(-x / math.sqrt(2))
    return sign * (spot * cdf(sign * d1) - strike * math.exp(-rate * maturity) * cdf(sign * d2))


def words(seed, path, group):
    """Independent integer implementation, checked against Random123's zero vector."""
    mask = (1 << 32) - 1
    c = [group & mask, group >> 32, path & mask, path >> 32]
    key = [seed & mask, seed >> 32]
    for _ in range(10):
        a, b = c[0] * 0xD2511F53, c[2] * 0xCD9E8D57
        c = [(b >> 32) ^ c[1] ^ key[0], b & mask, (a >> 32) ^ c[3] ^ key[1], a & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return c


def independent_asian(paths, steps, seed):
    """FP64 Python reference with separate path evolution and payoff code."""
    values = []
    for p in range(paths):
        z = []
        for group in range((steps + 3) // 4):
            w = words(seed, p, group)
            for i in (0, 2):
                a = ((w[i] >> 9) + .5) * 2 ** -23
                angle = ((w[i + 1] >> 9) + .5) * 2 ** -23 * 2 * math.pi
                radius = math.sqrt(-2 * math.log(a))
                z.extend((radius * math.cos(angle), radius * math.sin(angle)))
        value, total = 100., 0.
        for shock in z[:steps]:
            value *= math.exp((.03 - .2 ** 2 / 2) / steps + .2 / math.sqrt(steps) * shock)
            total += value
        values.append(math.exp(-.03) * max(total / steps - 100, 0))
    return sum(values) / paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exe = str(args.exe.resolve())
    checks = []
    base_o = dict(option_type="european_call", spot=100, strike=100, risk_free_rate=.03, volatility=.2, maturity=1)
    base_s = dict(num_paths=65536, num_steps=32, seed=1234, rng="curand", variance_reduction="none")

    def run(name, option=None, sim=None, fail=False):
        o, s = base_o | (option or {}), base_s | (sim or {})
        op, sp, out = (args.output / f"{name}{suffix}" for suffix in ("_option.cfg", "_sim.cfg", ""))
        config(op, o)
        config(sp, s)
        p = subprocess.run([exe, "--option", str(op), "--simulation", str(sp), "--backend",
                            "cpu" if args.cpu_only else "both", "--output", str(out)], text=True, capture_output=True)
        if fail:
            assert p.returncode != 0, name
            checks.append(name)
            return None
        assert p.returncode == 0, (name, p.stderr)
        data = json.loads(out.with_suffix(".json").read_text())
        cpu = data["cpu"]
        if not args.cpu_only:
            gpu = data["gpu"]
            for key in ("price", "standard_error", "delta", "gamma", "vega"):
                if cpu[key] is not None:
                    assert abs(cpu[key] - gpu[key]) < 3e-4 + abs(cpu[key]) * 1e-5, (name, key, cpu[key], gpu[key])
            assert cpu["independent_samples"] == gpu["independent_samples"]
        checks.append(name)
        return data

    assert words(0, 0, 0) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    checks.append("Philox zero known-answer vector (independent reference)")
    for kind in ("european_call", "european_put", "asian_call", "barrier_call"):
        for vr in ("none", "antithetic", "control_variate"):
            d = run(f"{kind}_{vr}", dict(option_type=kind), dict(variance_reduction=vr))
            r = d["cpu"]
            assert r["independent_samples"] == (32768 if vr == "antithetic" else 65536)
            if kind.startswith("european"):
                ref = bs(put=kind.endswith("put"))
                assert abs(r["reference_price"] - ref) < 1e-12
                assert abs(r["price"] - ref) < 5 * r["standard_error"] + 1e-4
            if kind == "asian_call" and vr == "control_variate":
                assert r["raw_to_adjusted_variance_ratio"] > 100
    for name, option in (("zero_vol", dict(volatility=0)), ("expiry", dict(maturity=0, spot=110)),
                         ("negative_rate", dict(risk_free_rate=-.02)), ("deep_itm", dict(spot=180)),
                         ("deep_otm", dict(spot=60))):
        d = run(name, option)
        p = base_o | option
        ref = bs(p["spot"], p["strike"], p["risk_free_rate"], p["volatility"], p["maturity"])
        assert abs(d["cpu"]["price"] - ref) < 5 * d["cpu"]["standard_error"] + 2e-5
    knocked = run("barrier_at_spot", dict(option_type="barrier_call", barrier=100))["cpu"]
    assert knocked["price"] == 0 and knocked["standard_error"] == 0
    deterministic = run("deterministic_asian", dict(option_type="asian_call", volatility=0), dict(num_steps=17))["cpu"]
    mean = sum(100 * math.exp(.03 * i / 17) for i in range(1, 18)) / 17
    assert abs(deterministic["price"] - math.exp(-.03) * max(mean - 100, 0)) < 3e-5
    last = None
    for block in (32, 96, 256, 512):
        d = run(f"layout_{block}", dict(option_type="asian_call"), dict(num_paths=10003, num_steps=17, block_size=block, blocks=7))
        current = d["cpu" if args.cpu_only else "gpu"]["price"]
        if last is not None:
            assert abs(current - last) < 1e-10
        last = current
    ref = independent_asian(37, 17, 2**48 + 19)
    d = run("independent_fp64_asian", dict(option_type="asian_call"), dict(num_paths=37, num_steps=17, seed=2**48 + 19))
    assert abs(d["cpu"]["price"] - ref) < 3e-5
    a = run("rng_curand", dict(option_type="asian_call"), dict(num_paths=10003, num_steps=17))
    b = run("rng_custom", dict(option_type="asian_call"), dict(num_paths=10003, num_steps=17, rng="custom"))
    for backend in ("cpu",) if args.cpu_only else ("cpu", "gpu"):
        assert abs(a[backend]["price"] - b[backend]["price"]) < 1e-10
    a = run("reproducible", dict(option_type="asian_call"), dict(num_paths=10003, num_steps=17))
    assert a["cpu"]["price"] == b["cpu"]["price"]
    one = run("one_step_asian", dict(option_type="asian_call"), dict(num_steps=1))["cpu"]
    euro = run("one_step_european", sim=dict(num_steps=1))["cpu"]
    assert abs(one["price"] - euro["price"]) < 1e-10
    greek = run("greeks", sim=dict(num_paths=500000))["cpu"]
    d1 = .25
    density = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    for key, expected, bias in (("delta", .5 * math.erfc(-d1 / math.sqrt(2)), 1e-4),
                                ("gamma", density / 20, 1e-4), ("vega", 100 * density, 1e-3)):
        assert abs(greek[key] - expected) < 5 * greek[key + "_standard_error"] + bias, (key, greek[key], expected)
    for name, o, s in (("bad_sigma", dict(volatility=-1), {}), ("nan", dict(spot="nan"), {}),
                       ("odd_antithetic", {}, dict(num_paths=9, variance_reduction="antithetic")),
                       ("bad_steps", {}, dict(num_steps=0)), ("overflow_steps", {}, dict(num_steps=2**40)),
                       ("bad_block", {}, dict(block_size=33)), ("bad_rng", {}, dict(rng="nonsense")),
                       ("unknown_key", {}, dict(num_path=1000)), ("negative_seed", {}, dict(seed=-1))):
        run(name, o, s, fail=True)
    summary = dict(status="PASS", checks=len(checks), cases=checks, cpu_only=args.cpu_only)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {len(checks)} validation cases")


if __name__ == "__main__":
    main()
