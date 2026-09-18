#!/usr/bin/env python3
"""Summarize repeated A/B timings, strict validation, and cache-policy NCU data.

Roofline uses useful FHT add/sub work, not executed dense MMA FLOPs. Its FP32
compute line is a non-TC reference ceiling, never a Tensor Core utilization claim.
Peak values must be explicitly supplied for the actual GPU; no cross-GPU mixing.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
import re
from pathlib import Path
import statistics


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def ncu_metrics(path):
    """NCU wide raw CSV: header, units row, one kernel measurement."""
    data = rows(path)
    if len(data) != 2:
        raise ValueError(f"expected one units row and one kernel: {path}")
    return {k: (v, data[0][k]) for k, v in data[1].items()}


def metric(metrics, key, target=None):
    if key not in metrics:
        return None
    value, unit = metrics[key]
    try:
        value = float(value.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    factors = {"s": 1, "second": 1, "ms": 1e-3, "msecond": 1e-3,
               "us": 1e-6, "usecond": 1e-6, "ns": 1e-9, "nsecond": 1e-9,
               "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
               "byte/block": 1, "Kbyte/block": 1e3, "Mbyte/block": 1e6,
               "byte/s": 1, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9,
               "Tbyte/s": 1e12}
    if target:
        if unit not in factors:
            raise ValueError(f"unknown unit {unit!r} for {key}")
        value *= factors[unit]
    return value


def profile_case(stem):
    match = re.fullmatch(r'(warp|tc_fast|tc_split)(?:_d(64|128|256|512|1024))?_(all|none)_raw', stem)
    if not match:
        raise ValueError(f'unknown NCU case name: {stem}')
    return match[1], int(match[2] or 128), match[3]


def summarize(timing_dir, verify_dir):
    timing = defaultdict(list)
    for path in sorted(timing_dir.glob("w*/timing_[123].csv")):
        for row in rows(path):
            key = (row['gpu'], path.parent.name, row['dtype'], int(row['head_dim']), row['implementation'])
            timing[key].append(float(row['avg_ms']) * 1000)
    results = []
    for (gpu, variant, dtype, dim, impl), values in sorted(timing.items()):
        results.append(dict(gpu=gpu, variant=variant, dtype=dtype, head_dim=dim,
                            implementation=impl, repeats=len(values), median_us=statistics.median(values),
                            min_us=min(values), max_us=max(values)))
    checks = []
    if verify_dir:
        for path in sorted(verify_dir.glob("w*/validation/validation.csv")):
            for row in rows(path):
                checks.append(dict(row, variant=path.parents[1].name))
    return results, checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing-dir', type=Path, required=True)
    p.add_argument('--verify-dir', type=Path)
    p.add_argument('--profile-dir', type=Path, help='Separate NCU job directory; defaults to verify-dir')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dram-peak-gbs', type=float)
    p.add_argument('--fp32-peak-tflops', type=float)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing, checks = summarize(args.timing_dir, args.verify_dir)
    if not timing:
        raise SystemExit('no timing data')
    if len({r['gpu'] for r in timing}) != 1:
        raise SystemExit('timing directory contains multiple GPUs; split before comparison')
    with (args.output_dir / 'timing_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    lines = ['# 严格验收与 A/B 调优结果', '',
             '时间为同一作业、相同输入的三次独立扫描中位数；每次 20 warmup / 100 CUDA Events。',
             '计时 GPU：' + ', '.join(sorted({r['gpu'] for r in timing})),
             f'计时来源：`{args.timing_dir}`；严格验证来源：`{args.verify_dir}`。',
             '验证与计数器可来自独立作业，其编译器和 GPU 见各自 environment.txt；不混算性能。', '',
             '| dtype | d | 路径 | 原配置 μs | 最快候选 | 中位数 μs | 加速 |',
             '|---|---:|---|---:|---|---:|---:|']
    for dtype in ('fp16', 'bf16'):
        for dim in (64, 128, 256, 512, 1024):
            for impl in ('warp', 'tc_fast', 'tc_split', 'fused_warp_int4', 'fused_tc_fast_int4', 'fused_tc_split_int4'):
                group = [r for r in timing if r['dtype'] == dtype and r['head_dim'] == dim and r['implementation'] == impl and r['repeats'] == 3]
                base = next((r for r in group if r['variant'] == 'w4_tc4_v0'), None)
                if not base:
                    continue
                best = min(group, key=lambda r: r['median_us'])
                lines.append(f"| {dtype} | {dim} | {impl} | {base['median_us']:.3f} | {best['variant']} | {best['median_us']:.3f} | {base['median_us']/best['median_us']:.3f}x |")
    lines += ['', '最快候选是已测参数中的事后比较，不等于可推广的自动分发策略。', '', '## 严格绝对误差与融合校验', '']
    if checks:
        lines += ['| 变体 | 参考 | 路径 | 通过/总数 | 最大绝对误差 |', '|---|---|---|---:|---:|']
        groups = defaultdict(list)
        for r in checks:
            groups[(r['variant'], r['reference'], r['implementation'])].append(r)
        for (variant, ref, impl), group in sorted(groups.items()):
            errors = [float(r['max_abs']) for r in group if r['max_abs']]
            lines.append(f"| {variant} | {ref} | {impl} | {sum(r['pass']=='True' for r in group)}/{len(group)} | {max(errors):.8g} |" if errors else
                         f"| {variant} | {ref} | {impl} | {sum(r['pass']=='True' for r in group)}/{len(group)} | n/a |")
        hashes = defaultdict(set)
        for r in checks:
            if r['sha256']:
                hashes[(r['case'], r['implementation'])].add(r['sha256'])
        differing = sum(len(v) > 1 for v in hashes.values())
        lines += ['', f'同一用例/后端在不同调优参数下输出 SHA256 不一致的组数：{differing}。',
                  f"独立检查器与 C++ 判据不一致的行数：{sum(r['harness_consistent']!='True' for r in checks)}。"]
    else:
        lines += ['暂无成功读取的严格验收产物；性能结果不能替代正确性验收。']

    counters = []
    profile_dir = args.profile_dir or args.verify_dir
    if profile_dir:
        for path in sorted(profile_dir.glob('w*/ncu/*_raw.csv')):
            m = ncu_metrics(path)
            duration = metric(m, 'gpu__time_duration.avg', 'seconds')
            rate = metric(m, 'dram__bytes.sum.per_second', 'bytes/s')
            byte_count = metric(m, 'dram__bytes.sum', 'bytes')
            source = 'dram__bytes.sum'
            if byte_count is None and duration is not None and rate is not None:
                byte_count = duration * rate
                source = 'NCU DRAM rate * NCU duration'
            if not duration or byte_count is None:
                continue
            impl, dim, policy = profile_case(path.stem)
            useful_ops = 16384 * dim * int(math.log2(dim))  # scale excluded
            counters.append(dict(gpu=m.get('device__attribute_display_name', ('unknown', ''))[0],
                variant=path.parents[1].name, implementation=impl, cache_control=policy,
                head_dim=dim, total_tokens=16384,
                duration_us=duration*1e6, dram_bytes=byte_count, byte_source=source,
                useful_gflops=useful_ops/duration/1e9, useful_flops_per_dram_byte=useful_ops/byte_count if byte_count else None,
                tensor_active_pct=metric(m, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
                short_scoreboard=metric(m, 'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),
                mio_throttle=metric(m, 'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio')))
    if counters:
        profile_gpus = sorted({r['gpu'] for r in counters})
        if len(profile_gpus) != 1:
            raise SystemExit('NCU directory contains multiple GPUs; split before plotting')
        with (args.output_dir / 'cache_ncu_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(counters[0])); writer.writeheader(); writer.writerows(counters)
        lines += ['', '## NCU 缓存策略与 Roofline', '',
                  'NCU GPU：' + ', '.join(profile_gpus) + f'；共 {len(counters)} 次单 kernel 采集。',
                  f'独立计数器目录：`{profile_dir}`。其 GPU/环境见该目录的 environment.txt；不与计时作业跨 GPU 混算。',
                  '`all` 表示 replay 前刷新，`none` 表示不由 profiler 刷新，不直接等同于应用 cold/warm。',
                  '横轴使用实际 DRAM 字节（或明确标注的 NCU rate×duration），不是逻辑张量字节。',
                  '纵轴统一使用有用 FHT 加减次数；TC 的冗余 MMA FLOPs 不算成算法收益。',
                  'FP32 roof 仅为非 TC 参考线，不是 Tensor Core 峰值或利用率。',
                  '![缓存策略 Roofline](roofline_cache.png)']
        lines += ['', '单次 profiler replay，不能与上面的 131072 tokens 无 profiler 中位数混算。',
                  '以下固定 FP16、normalize=true、16384 tokens、cache-control=all；stall 是',
                  '`per_issue_active.ratio`（每次发射对应的 stalled warp 周期），不是百分比。', '',
                  '| 配置 | 路径 | d | NCU μs | short scoreboard | MIO throttle | tensor active % |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for r in counters:
            if r['variant'] in ('w4_tc4_v0', 'w0_tc4_v1') and r['cache_control'] == 'all':
                vals = ['n/a' if r[k] is None else f'{r[k]:.3f}' for k in
                        ('duration_us', 'short_scoreboard', 'mio_throttle', 'tensor_active_pct')]
                lines.append(f"| {r['variant']} | {r['implementation']} | {r['head_dim']} | " + ' | '.join(vals) + ' |')
    lines += ['', '![重复扫描性能对比](tuning.png)', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; tables written, figures not generated')
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    variants = sorted({r['variant'] for r in timing})
    for axis, dtype, impl in zip(axes.flat, ('fp16','bf16','fp16','bf16'), ('warp','warp','tc_fast','tc_fast')):
        for variant in variants:
            group = [r for r in timing if r['dtype']==dtype and r['implementation']==impl and r['variant']==variant]
            group.sort(key=lambda r: r['head_dim'])
            if not group: continue
            axis.errorbar([str(r['head_dim']) for r in group], [r['median_us'] for r in group],
                          yerr=[[r['median_us']-r['min_us'] for r in group], [r['max_us']-r['median_us'] for r in group]],
                          marker='o', label=variant)
        axis.set(title=f'{dtype} / {impl}', xlabel='head_dim', ylabel='Median kernel time (us)', yscale='log')
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / 'tuning.png', dpi=180); plt.close(fig)
    if counters:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
        for axis, policy in zip(axes, ('all','none')):
            subset = [r for r in counters if r['cache_control']==policy and r['variant'] in ('w4_tc4_v0','w0_tc4_v1')]
            zero_traffic = sum(r['useful_flops_per_dram_byte'] is None for r in subset)
            for row in subset:
                if row['useful_flops_per_dram_byte'] is None:
                    continue
                color = {'warp':'tab:blue','tc_fast':'tab:orange','tc_split':'tab:green'}[row['implementation']]
                if row['implementation'] == 'warp' and row['head_dim'] == 64:
                    color = 'tab:purple'
                marker = 'x' if row['variant']=='w4_tc4_v0' else 'o'
                label = ('base/' if marker=='x' else 'tuned/') + row['implementation'] + f"/d{row['head_dim']}"
                axis.scatter(row['useful_flops_per_dram_byte'], row['useful_gflops'], s=40,
                             color=color, marker=marker, label=label)
            if zero_traffic:
                axis.text(.03, .97, f'{zero_traffic} captures: zero DRAM bytes; intensity undefined',
                          transform=axis.transAxes, va='top', fontsize=8)
            xs = [10**(-1+i/30) for i in range(121)]
            if args.dram_peak_gbs and args.fp32_peak_tflops:
                axis.plot(xs, [min(x*args.dram_peak_gbs, args.fp32_peak_tflops*1000) for x in xs], '--', color='gray', label='HBM + FP32 reference roof (not TC peak)')
            axis.set(xscale='log', yscale='log', xlabel='Useful FHT FLOP / measured DRAM byte',
                     ylabel='Useful FHT GFLOP/s', title=f'{profile_gpus[0]} / cache-control={policy}')
            axis.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, .15, 1, 1))
        fig.savefig(args.output_dir / 'roofline_cache.png', dpi=180); plt.close(fig)
    (args.output_dir / 'plot_metadata.json').write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == '__main__':
    main()
