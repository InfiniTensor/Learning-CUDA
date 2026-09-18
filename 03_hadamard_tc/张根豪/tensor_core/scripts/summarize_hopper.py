#!/usr/bin/env python3
"""Same-GPU Hopper counters and independent wide-store repetitions."""
import argparse
import csv
import math
from pathlib import Path
import re
from summarize_audit import ncu_metrics, metric
from summarize_library import aggregate
from roofline_plot import plot_roofline


def counters(directory):
    rows = []
    for path in sorted((directory/'ncu').glob('*_raw.csv')):
        match = re.fullmatch(r'(warp|wmma_fast|wmma_split|ptx_fast|ptx_split)_d(64|128|256)_(all|none)_raw', path.stem)
        if not match: raise ValueError(f'unknown capture: {path}')
        backend, dim, policy = match.groups(); dim = int(dim)
        m = ncu_metrics(path)
        duration = metric(m, 'gpu__time_duration.avg', 'seconds')
        byte_count = metric(m, 'dram__bytes.sum', 'bytes')
        source = 'dram__bytes.sum'
        if byte_count is None:
            rate = metric(m, 'dram__bytes.sum.per_second', 'bytes/s')
            if rate is None or not duration: raise ValueError(f'missing counters: {path}')
            byte_count = rate * duration
            source = 'NCU DRAM rate * NCU duration'
        if not duration or byte_count < 0: raise ValueError(f'invalid counters: {path}')
        ops = 16384 * dim * int(math.log2(dim))
        rows.append(dict(gpu=m.get('device__attribute_display_name', ('unknown',''))[0],
            backend=backend, dim=dim, cache_control=policy, duration_us=duration*1e6,
            dram_bytes=byte_count, byte_source=source, useful_gflops=ops/duration/1e9,
            useful_flops_per_dram_byte=ops/byte_count if byte_count else None,
            static_shared=metric(m,'launch__shared_mem_per_block_static','bytes'),
            dynamic_shared=metric(m,'launch__shared_mem_per_block_dynamic','bytes'),
            registers=metric(m,'launch__registers_per_thread'),
            short_scoreboard=metric(m,'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),
            mio_throttle=metric(m,'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'),
            tensor_active_pct=metric(m,'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')))
    if len({r['gpu'] for r in rows}) > 1: raise ValueError('multiple GPUs in NCU directory')
    return rows


def quant_repeats(directory):
    result = []
    for path in sorted(directory.glob('quant_repeat_*/timings.csv')):
        rows = aggregate(path)
        index = {tuple(r[k] for k in ('dtype','tokens','head_dim','normalize','backend')):r for r in rows}
        for r in rows:
            if '/' not in r['backend']: continue
            key = tuple(r[k] for k in ('dtype','tokens','head_dim','normalize'))
            base = index[key+('f4_q0/fused_warp',)]
            result.append(dict(r,process=path.parent.name,control_us=base['median_us'],
                               speedup=base['median_us']/r['median_us']))
    if len({r['gpu'] for r in result}) > 1: raise ValueError('multiple GPUs in repetition data')
    return result


def save_csv(path, rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--profile-dir',type=Path,help='Original NCU directory when captures are shared by reports')
    p.add_argument('--dram-peak-gbs',type=float,required=True)
    p.add_argument('--fp32-peak-tflops',type=float,required=True)
    args=p.parse_args()
    data=counters(args.profile_dir or args.directory); repeats=quant_repeats(args.directory)
    timing=aggregate(args.directory/'benchmark/timings.csv')
    gpus={r['gpu'] for r in data+repeats+timing}
    if len(gpus)!=1: raise ValueError(f'cross-GPU comparison prohibited: {gpus}')
    if len(data)!=30: raise ValueError(f'expected 30 captures, got {len(data)}')
    if len(repeats)!=72 or any(r['repeats']!=9 or not r['all_pass'] for r in repeats):
        raise ValueError('expected three complete, passing 9-round quant repetitions')
    save_csv(args.directory/'ncu_summary.csv',data)
    save_csv(args.directory/'quant_repeats.csv',repeats)
    save_csv(args.directory/'timing_summary.csv',timing)
    lines=['# Hopper 缺口补测','',f'GPU：{next(iter(gpus))}。30 次 NCU 成功采集；量化为三个独立进程，每进程九轮交错。','',
      '## NCU：同卡、同输入、单 kernel','',
      'FP16，16384 tokens，normalize=true；随机输入避免特殊值内存压缩偏置。',
      '四次 launch 取第四次；cache-control=all/none；clock-control=none，未锁时钟。',
      '下表为 all 策略，时间 μs、shared 字节；stall 是每 active issue 的比值，不是百分比。','',
      '| d | 路径 | μs | dynamic shared | registers | short scoreboard | MIO throttle |',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in data:
        if r['cache_control']=='all':
            lines.append(f"| {r['dim']} | {r['backend']} | {r['duration_us']:.3f} | {r['dynamic_shared']} | {r['registers']} | {r['short_scoreboard']} | {r['mio_throttle']} |")
    lines+=['','![Hopper NCU Roofline](roofline.png)','',
      'Roofline 工作量为有用 FHT 加减次数 N·d·log2(d)，不计 dense MMA 冗余操作。',
      '横轴使用实际 DRAM 流量（缺少 bytes 时使用同次 NCU rate×duration，CSV 标明来源）。',
      'FP32 参考 roof 不是 Tensor Core 峰值；all/none 不等于真实应用的冷/热缓存。',
      '未刷新缓存时可命中 L2，DRAM 点可越过 DRAM 斜线；不代表超越硬件带宽。','',
      '## 同卡大规模性能','',
      '131072 tokens，normalize=true，五轮 Graph 摊销设备时间中位数，μs。',
      '不与单次 NCU 时间混算；TC 仍须单独看严格正确性，快不等于可用于提交。','',
      '| dtype | d | warp | WMMA fast | PTX fast | WMMA split | PTX split |',
      '|---|---:|---:|---:|---:|---:|---:|']
    modes=('warp','wmma_fast','mma_fast','wmma_split','mma_split')
    for dt in ('float16','bfloat16'):
        for d in (64,128,256):
            group={r['backend'].split('/')[-1]:r['median_us'] for r in timing if r['dtype']==dt and int(r['head_dim'])==d and r['tokens']=='131072' and r['normalize']=='1'}
            lines.append(f'| {dt} | {d} | '+' | '.join(f'{group[m]:.3f}' for m in modes)+' |')
    lines+=['','![Hopper 同卡性能](ptx_comparison.png)','','## INT4 宽写回独立复测','',
      '对照固定 f4_q0；f4_q1 隔离写回宽度，f8_q1 同时改变融合 block 大小。',
      '表内为三个进程各自九轮中位数的 speedup，不挑最快轮；全部输出检查通过。','',
      '| dtype | tokens | d | 候选 | 进程 1 / 2 / 3 加速 | 三次均 >1 |',
      '|---|---:|---:|---|---|---|']
    for dt in ('float16','bfloat16'):
        for n in (16384,131072):
            for d in (512,1024):
                for candidate in ('f4_q1/fused_warp','f8_q1/fused_warp'):
                    group=sorted([r for r in repeats if r['dtype']==dt and int(r['tokens'])==n and int(r['head_dim'])==d and r['backend']==candidate],key=lambda r:r['process'])
                    values=[r['speedup'] for r in group]
                    lines.append(f'| {dt} | {n} | {d} | {candidate} | '+ ' / '.join(f'{v:.3f}×' for v in values)+f' | {all(v>1 for v in values)} |')
    lines+=['','![独立进程复测](quant_repeat.png)','']
    (args.directory/'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_roofline(data, args.directory/'roofline.png', args.dram_peak_gbs, args.fp32_peak_tflops)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dt in zip(axes,('float16','bfloat16')):
        for mode in modes:
            rows=sorted([r for r in timing if r['dtype']==dt and r['tokens']=='131072' and r['normalize']=='1' and r['backend'].endswith('/'+mode)],key=lambda r:int(r['head_dim']))
            ax.plot([int(r['head_dim']) for r in rows],[r['median_us'] for r in rows],marker='o',label=mode)
        ax.set(title=dt,xlabel='head_dim',ylabel='Graph amortized device time (us)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.directory/'ptx_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dt in zip(axes,('float16','bfloat16')):
        for d in (512,1024):
            for candidate in ('f4_q1/fused_warp','f8_q1/fused_warp'):
                rows=sorted([r for r in repeats if r['dtype']==dt and int(r['head_dim'])==d and r['tokens']=='131072' and r['backend']==candidate],key=lambda r:r['process'])
                ax.plot([1,2,3],[r['speedup'] for r in rows],marker='o',label=f'd{d}/{candidate.split("/")[0]}')
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'{dt}, 131072 tokens',xlabel='Independent process',ylabel='f4_q0 / candidate');ax.set_xticks([1,2,3]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.directory/'quant_repeat.png',dpi=180);plt.close(fig)


if __name__=='__main__': main()
