#!/usr/bin/env python3
"""Acceptance and honest performance cost of guarded TC / selective INT4."""
import argparse
import csv
from pathlib import Path
from summarize_library import aggregate


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    root=args.directory
    guard=read(root/'guard_validation/validation.csv')
    quant=read(root/'quant_validation/validation.csv')
    stress=read(root/'stress/stress.csv')
    if len(guard)!=216 or len(quant)!=528 or len(stress)!=192:
        raise ValueError('incomplete acceptance matrix')
    if any(r['passed']!='True' for r in guard+quant+stress):
        raise ValueError('acceptance failure; inspect raw CSV before promotion')
    timing=aggregate(root/'guard_benchmark/timings.csv')
    qt=aggregate(root/'quant_benchmark/timings.csv')
    if len({r['gpu'] for r in timing+qt})!=1: raise ValueError('mixed GPUs')
    lines=['# 保护式 TC 与 D1024 选择性宽写回','',
      'GPU：'+timing[0]['gpu']+'；同一作业，所有 A/B 在同一卡上执行。',
      '保护式 TC 指 MMA_PRECISION=3 的 PTX split + warp 回退，不是纯 TC。','',
      '## 严格输出检查','',
      '- 保护式 TC 与 warp 对照：各108/108核心配置。',
      '- 保护式 TC 幅度/相消压力测试：192/192，每配置512 tokens。',
      '- 选择性宽写回及对照：2参数 × 132配置 × 2后端 = 528行全部通过。',
      '- 判据未变：FP16 <0.01、BF16 <0.05；融合 packed INT4 与 scale 逐位比较。',
      '- 当前测试通过不构成任意位模式或任意 GPU 的形式化证明。','',
      '## 保护式 TC 的真实成本','',
      '131072 tokens、normalize=true，五轮 Graph 摊销设备时间中位数，μs。',
      '计时包含保护分支及触发时的回退重算，不能仅引用未保护 PTX 的更快数字。','',
      '| dtype | d | warp | 保护式 TC | warp / 保护式 TC |', '|---|---:|---:|---:|---:|']
    ratios=[]
    for dtype in ('float16','bfloat16'):
        for dim in (64,128,256):
            rows={r['backend']:r for r in timing if r['dtype']==dtype and int(r['head_dim'])==dim and r['tokens']=='131072' and r['normalize']=='1'}
            w,t=rows['guard/warp']['median_us'],rows['guard/mma_split']['median_us']
            lines.append(f'| {dtype} | {dim} | {w:.3f} | {t:.3f} | {w/t:.3f}× |')
            ratios.append((dtype,dim,w/t))
    lines+=['','## 选择性宽写回','',
      'D1024 使用已验证的128-bit写回，D512保留原写回；其余维度不变。',
      '131072 tokens、normalize=true，九轮交错中位数，μs。','',
      '| dtype | d | q0 对照 | q2 选择性 | 对照 / 选择性 |', '|---|---:|---:|---:|---:|']
    qratios=[]
    for dtype in ('float16','bfloat16'):
        for dim in (512,1024):
            rows={r['backend']:r for r in qt if r['dtype']==dtype and int(r['head_dim'])==dim and r['tokens']=='131072' and r['normalize']=='1'}
            a,b=rows['control/fused_warp']['median_us'],rows['selective/fused_warp']['median_us']
            lines.append(f'| {dtype} | {dim} | {a:.3f} | {b:.3f} | {a/b:.3f}× |')
            qratios.append((dtype,dim,a/b))
    lines+=['','![保护成本与选择性写回](guard_quant_comparison.png)','',
      '默认保持 warp，不自动替换为 TC；选择性写回可用 QUANT_VECTOR=2 显式启用。',
      'memcheck/racecheck/synccheck 的原始日志在本目录；不能把通过正确性当作性能收益。']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,values,title in zip(axes,(ratios,qratios),('warp / guarded TC','q0 / selective q2')):
        for dtype in ('float16','bfloat16'):
            points=[(d,r) for dt,d,r in values if dt==dtype]
            ax.plot([d for d,r in points],[r for d,r in points],marker='o',label=dtype)
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=title,xlabel='head_dim',ylabel='Speed ratio (>1 favors candidate)')
        ax.set_xticks(sorted({d for _,d,_ in values}))
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'guard_quant_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__': main()
