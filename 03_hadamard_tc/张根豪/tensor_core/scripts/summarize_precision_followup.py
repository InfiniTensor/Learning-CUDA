#!/usr/bin/env python3
"""Report strict acceptance, newly fixed/regressed cases and precision cost."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from summarize_library import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    with (args.directory/'precision_validation/validation.csv').open() as f:
        checks=list(csv.DictReader(f))
    timing=aggregate(args.directory/'precision_benchmark/timings.csv')
    groups=defaultdict(list)
    for r in checks: groups[(r['variant'],r['backend'])].append(r)
    if len(checks)!=648 or any(len(rows)!=108 for rows in groups.values()):
        raise ValueError('expected 3 variants x 2 backends x 108 cases')
    lines=['# PTX split 精度改进实测','',
      '实际 Dao v1.1.0，全张量 native 最大绝对误差；FP16 <0.01，BF16 <0.05。',
      '每组核心矩阵为108配置，不能与包含D512/1024的132配置分母混用。',
      'p0 原算法；p1 高低部分独立累加；p2 再增加一层 native 残差。','',
      '| 参数 | 后端 | 严格通过/总数 | 最大绝对误差 |', '|---|---|---:|---:|']
    for (variant,backend),rows in sorted(groups.items()):
        lines.append(f"| {variant} | {backend} | {sum(r['passed']=='True' for r in rows)}/108 | {max(float(r['max_abs']) for r in rows):.8g} |")
    lines+=['','## 相对 p0：修复与回归','',
            '| 候选 | 原失败变通过 | 原通过变失败 | 输出哈希不同 |', '|---|---:|---:|---:|']
    baseline={r['case']:r for r in groups[('p0','mma_split')]}
    transitions=[]
    for variant in ('p1','p2'):
        rows=groups[(variant,'mma_split')]
        fixed=sum(r['passed']=='True' and baseline[r['case']]['passed']!='True' for r in rows)
        regressed=sum(r['passed']!='True' and baseline[r['case']]['passed']=='True' for r in rows)
        changed=sum(r['sha256']!=baseline[r['case']]['sha256'] for r in rows)
        lines.append(f'| {variant} | {fixed} | {regressed} | {changed} |')
        for r in rows:
            if r['passed']!=baseline[r['case']]['passed']:
                transitions.append(dict(r,original_passed=baseline[r['case']]['passed'],original_max_abs=baseline[r['case']]['max_abs']))
    if transitions:
        with (args.directory/'precision_transitions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(transitions[0]));w.writeheader();w.writerows(transitions)
    lines+=['','## 性能成本','',
      '131072 tokens，normalize=true，五轮 Graph 摊销设备时间中位数，μs；',
      '同一 GPU 和输入，不能用性能代替严格验收。','',
      '| dtype | d | p0 split | p1 split | p2 split | warp |', '|---|---:|---:|---:|---:|---:|']
    for dtype in ('float16','bfloat16'):
        for dim in (64,128,256):
            index={r['backend']:r['median_us'] for r in timing if r['dtype']==dtype and int(r['head_dim'])==dim and r['tokens']=='131072' and r['normalize']=='1'}
            lines.append(f'| {dtype} | {dim} | '+' | '.join(f'{index[k]:.3f}' for k in ('p0/mma_split','p1/mma_split','p2/mma_split','p0/warp'))+' |')
    all_pass=[v for v in ('p0','p1','p2') if all(r['passed']=='True' for r in groups[(v,'mma_split')])]
    lines+=['', '本矩阵全面通过的 split 参数：'+(', '.join(all_pass) if all_pass else '无')+'。',
            '即使本矩阵全过也不是任意输入的数学证明；仍需扩大输入范围、验证稳定收益。',
            '未放宽 PDF 判据、未切换默认。失败配置详见 precision_validation/status.json。']
    (args.directory/'precision_summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
