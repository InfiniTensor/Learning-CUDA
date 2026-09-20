#!/usr/bin/env python3
import csv, json, pathlib, statistics, sys, html
run=pathlib.Path(sys.argv[1]); groups=[]
for p in sorted((run/'csv').glob('*.csv')):
 with p.open() as f: rows=list(csv.DictReader(f))
 if not rows: raise RuntimeError('empty CSV '+str(p))
 row={'case':p.stem,'variant':rows[0]['variant'],'format':int(rows[0]['format']),'n':int(rows[0]['n']),'input_dtype':int(rows[0]['input_dtype'])}
 for k in rows[0]:
  if k in ['variant','iteration']: continue
  values=[float(r[k]) for r in rows]
  row[k+'_mean']=statistics.mean(values); row[k+'_std']=statistics.stdev(values) if len(values)>1 else 0
 row['e2e_speedup']=row['cpu_e2e_ms_mean']/row['e2e_ms_mean']
 groups.append(row)
(run/'summary.json').write_text(json.dumps(groups,indent=2))
with (run/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
lines=['# 实测汇总','', '全部由原始 CSV 自动生成。时间单位 ms；±为样本标准差。CPU 单线程 -O3。', '', '|case|量化|反量化|含传输处理|GPU文件E2E|CPU/GPU处理加速|文件E2E加速|MAE|payload压缩|', '|---|---|---|---|---|---|---|---|---|']
for g in groups:
 lines.append('|'+g['case']+'|'+ '|'.join(f"{g[k+'_mean']:.5g} ± {g[k+'_std']:.2g}" for k in ['quant_ms','dequant_ms','process_ms','e2e_ms'])+f"|{g['process_speedup_mean']:.3g}|{g['e2e_speedup']:.3g}|{g['mae_mean']:.4g}|{g['payload_ratio_mean']:.4g}|")
(run/'summary.md').write_text('\n'.join(lines)+'\n')
# 使用标准绘图库从 CSV 生成可导出 SVG/PNG。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
items=[g for g in groups if '_fp32_normal_' in g['case']]
fig,ax=plt.subplots(figsize=(12,max(4,len(items)*.3)))
ax.barh([g['case'] for g in items],[g['quant_ms_mean']+g['dequant_ms_mean'] for g in items],color=['#157a6e' if g['variant']=='optimized' else '#d88035' for g in items])
ax.set_xlabel('Quantization + dequantization CUDA event time (ms)')
ax.set_title('FP32 normal input, seed=42; CPU/GPU validated')
fig.tight_layout()
for ext in ['svg','png']:fig.savefig(run/('performance.'+ext),dpi=160)
plt.close(fig)
with (run/'quality.csv').open() as f:q=[r for r in csv.DictReader(f) if r['input_dtype']=='0' and r['output_dtype']=='0' and r['mode']=='0' and r['round']=='0']
fig,ax=plt.subplots(figsize=(10,5))
labels=[['MX-E4M3','MX-E5M2','NV-E2M1'][int(r['format'])]+' '+r['distribution'] for r in q]
ax.barh(labels,[float(r['mae']) for r in q]);ax.set_xlabel('Reconstruction MAE');ax.set_title('FP32 input/output, block, nearest, seed=42')
fig.tight_layout()
for ext in ['svg','png']:fig.savefig(run/('error.'+ext),dpi=160)
plt.close(fig)
print('summarized',len(groups),'cases')
