"""Portable CUDA-event timer: 5 warmups, cold L2, discard one, 31 samples.

Extracted from the local kernel-benchmark timing helper used by the recorded runs.
"""

import statistics
import torch


def _summarize_times(times):
    if not times:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p20": 0.0,
            "p80": 0.0,
            "num_trials": 0,
        }
    vals = sorted(float(v) for v in times)

    def pct(p):
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p20": pct(0.20),
        "p80": pct(0.80),
        "num_trials": len(vals),
    }


def _clear_l2_cache_torch(device):
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def _bench_times_cuda_event(fn, cfg):
    device = torch.device(f"cuda:{cfg.device}")
    times = []
    with torch.cuda.device(device):
        for _ in range(cfg.num_warmup):
            fn()
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        for trial in range(cfg.num_trials + cfg.discard_first):
            torch.cuda.synchronize(device=device)
            _clear_l2_cache_torch(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device=device)
            if trial >= cfg.discard_first:
                times.append(float(start.elapsed_time(end)))
    return times
