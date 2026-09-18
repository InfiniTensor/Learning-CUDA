#!/usr/bin/env python3
"""Combine experimental PTX correctness, graph timing and optional NCU counters."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from summarize_audit import ncu_metrics,metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    with (args.directory/'validation/validation.csv').open() as f:checks=list(csv.DictReader(f))
    with (args.directory/'benchmark/summary.csv').open() as f:timing=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in checks:groups[r['backend']].append(r)
    lines=['# 显式 PTX 寄存器重排实验','',
           'GPU：'+', '.join(sorted({r['gpu'] for r in timing})),
           '未接入默认分发；仅 d=64/128/256 变换，尾部为 warp 回退。',
           'PTX 不使用 shared memory，但保留 fast/split 原数值算法，不代表修复严格精度失败。','',
           '| 路径 | 严格通过/总数 | 最大绝对误差 |','|---|---:|---:|']
    for name,rows in groups.items():
        lines.append(f"| {name} | {sum(r['passed']=='True' for r in rows)}/{len(rows)} | {max(float(r['max_abs']) for r in rows):.8g} |")
    for mode in ('fast','split'):
        a={r['case']:r['sha256'] for r in groups['mma_'+mode]}
        b={r['case']:r['sha256'] for r in groups['wmma_'+mode]}
        lines+=['',f'{mode} PTX/WMMA 输出哈希不同的配置数：{sum(a[k]!=b[k] for k in a)}。']
    lines+=['','## 大规模归一化计时','',
            '131072 tokens；同一进程、Graph 回放摊销设备时间，五轮中位数。单位 μs。','',
            '| dtype | d | warp | WMMA fast | PTX fast | WMMA split | PTX split |',
            '|---|---:|---:|---:|---:|---:|---:|']
    modes=('warp','wmma_fast','mma_fast','wmma_split','mma_split')
    for dtype in ('float16','bfloat16'):
        for dim in (64,128,256):
            group={r['backend'].split('/')[-1]:float(r['median_us']) for r in timing if r['dtype']==dtype and int(r['head_dim'])==dim and r['tokens']=='131072' and r['normalize']=='1'}
            lines.append(f'| {dtype} | {dim} | '+' | '.join(f'{group[m]:.3f}' for m in modes)+' |')
    lines+=['','![规模与 dtype 下的 PTX/WMMA 对照](mma_comparison.png)','','## NCU 独立诊断','']
    metrics=[]
    for path in sorted((args.directory/'ncu').glob('*_raw.csv')):
        data=ncu_metrics(path)
        duration=metric(data,'gpu__time_duration.avg','seconds')
        metrics.append(dict(backend=path.stem,us=duration*1e6 if duration else None,
            shared_static=metric(data,'launch__shared_mem_per_block_static','bytes'),
            shared_dynamic=metric(data,'launch__shared_mem_per_block_dynamic','bytes'),
            registers=metric(data,'launch__registers_per_thread'),
            short_scoreboard=metric(data,'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'),
            mio_throttle=metric(data,'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio')))
    if metrics:
        with (args.directory/'ncu_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
        lines+=['FP16 d=128、16384 tokens、normalize=true、cache-control=all，未锁时钟；',
                '该单次 profiler 时间不与上面的 Graph 时间混算。详见 ncu_summary.csv。']
    else:
        lines+=['没有成功导出的 NCU 原始 CSV；失败日志保留，不能伪造硬件计数器。',
                '共享内存/寄存器静态资源另见 resources.txt，不等于运行时 NCU 指标。']
    (args.directory/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(dtype,tokens) in zip(axes.flat,[(dt,n) for dt in ('float16','bfloat16') for n in (128,131072)]):
        for mode in modes:
            rows=[r for r in timing if r['dtype']==dtype and int(r['tokens'])==tokens and r['normalize']=='1' and r['backend'].endswith('/'+mode)]
            rows.sort(key=lambda r:int(r['head_dim']))
            ax.plot([int(r['head_dim']) for r in rows],[float(r['median_us']) for r in rows],marker='o',label=mode)
        ax.set(title=f'{dtype}, {tokens} tokens',xlabel='head_dim',ylabel='Graph amortized time (us)',yscale='log')
        ax.set_xticks([64,128,256])
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(', '.join(sorted({r['gpu'] for r in timing}))+' — normalized CUDA Graph comparison')
    fig.tight_layout();fig.savefig(args.directory/'mma_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
