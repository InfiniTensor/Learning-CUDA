#!/usr/bin/env python3
"""Report graph-resident library comparisons without mixing eager/kernel times."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def aggregate(path):
    groups = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['measurement'] != 'graph_amortized_device':
                raise ValueError('mixed measurement boundaries')
            key = tuple(row[k] for k in ('gpu','dtype','tokens','head_dim','normalize','backend'))
            groups[key].append(row)
    if len({key[0] for key in groups}) > 1:
        raise ValueError('multiple GPUs in one comparison file')
    result = []
    for key, records in sorted(groups.items()):
        if len({r['round'] for r in records}) != len(records):
            raise ValueError('duplicate rounds')
        if len({(r.get('nodes'),r.get('replays')) for r in records}) != 1:
            raise ValueError('mixed graph sizes or replay counts')
        values = [float(r['us']) for r in records]
        result.append(dict(zip(('gpu','dtype','tokens','head_dim','normalize','backend'),key),
            median_us=statistics.median(values), min_us=min(values), max_us=max(values),
            repeats=len(values), all_pass=all(r['pass']=='True' for r in records)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--plot-variant', help='Limit figure to one variant; all variants remain in tables')
    args = p.parse_args()
    data = aggregate(args.directory/'timings.csv')
    if not data: raise SystemExit('empty timing file')
    with (args.directory/'summary.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    index = {(r['dtype'],r['tokens'],r['head_dim'],r['normalize'],r['backend']):r for r in data}
    lines=['# 参考库同边界性能对照','',
           'GPU：'+', '.join(sorted({r['gpu'] for r in data})),
           '同一输入驻留 GPU，CUDA Graph 回放按节点数摊销的设备时间；不是 Python API 延迟，',
           '也不是独立 profiler 单 kernel 时间。捕获/分配不计时；固定地址缓存状态由规模决定。',
           '多轮交错后端顺序，中位数/min/max 均保留于 summary.csv。',
           '融合对照为 Dao 变换 + 本项目同协议 standalone INT4，不声称 Dao 自带 INT4。', '',
           '| dtype | tokens | d | normalize | 路径 | 本项目 μs | 对照 μs | 对照/本项目 | 全部检查通过 |',
           '|---|---:|---:|---:|---|---:|---:|---:|---|']
    comparisons=[]
    for r in data:
        if '/' not in r['backend']: continue
        quant='fused' in r['backend']
        ref=index[tuple(r[k] for k in ('dtype','tokens','head_dim','normalize'))+('dao_plus_quant' if quant else 'dao',)]
        speedup=ref['median_us']/r['median_us']
        comparisons.append(dict(r,reference_us=ref['median_us'],speedup=speedup))
        lines.append(f"| {r['dtype']} | {r['tokens']} | {r['head_dim']} | {r['normalize']} | {r['backend']} | {r['median_us']:.3f} | {ref['median_us']:.3f} | {speedup:.3f}× | {r['all_pass']} |")
    lines += ['', '![规模与参考库相对性能](library_comparison.png)', '']
    (args.directory/'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(dtype,impl) in zip(axes.flat,[(d,i) for d in ('float16','bfloat16') for i in ('warp','fused_warp')]):
        for backend in sorted({r['backend'] for r in comparisons if r['backend'].split('/')[-1]==impl}):
            if args.plot_variant and backend.split('/')[0] != args.plot_variant: continue
            for dim in (64,128,256,512,1024):
                group=[r for r in comparisons if r['dtype']==dtype and r['backend']==backend and int(r['head_dim'])==dim and r['normalize']=='1']
                group.sort(key=lambda r:int(r['tokens']))
                if group: ax.plot([int(r['tokens']) for r in group],[r['speedup'] for r in group],marker='o',label=f'{backend}/d{dim}')
        ax.axhline(1,color='gray',linestyle='--')
        ax.set(title=f'{dtype}/{impl}, normalized',xlabel='tokens',ylabel='Reference / our device time',xscale='log')
        ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(args.directory/'library_comparison.png',dpi=180); plt.close(fig)


if __name__=='__main__': main()
