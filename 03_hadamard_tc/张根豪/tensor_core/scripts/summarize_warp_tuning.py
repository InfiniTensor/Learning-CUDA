#!/usr/bin/env python3
"""Plot all repeated candidate timings and retain generated store-instruction evidence."""
import argparse
import csv
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--sass-dir',type=Path,required=True)
    args=p.parse_args()
    with (args.directory/'benchmark/summary.csv').open() as f: data=list(csv.DictReader(f))
    stores=[]
    for file in sorted(args.sass_dir.glob('w*.sass')):
        for body in file.read_text().split('Function : ')[1:]:
            name=body.splitlines()[0]
            match=re.search(r'Li(512|1024)ELb1E',name)
            if 'hadamard_warp_fused_int4_kernel' not in name or not match: continue
            stores.append(dict(variant=file.stem,dtype='fp16' if '__half' in name else 'bf16',
                dim=int(match[1]),byte_stores=len(re.findall(r'\bSTG\.E\.U8\b',body)),
                stores_64=len(re.findall(r'\bSTG\.E\.64\b',body)),
                stores_128=len(re.findall(r'\bSTG\.E\.128\b',body))))
    if stores:
        with (args.directory/'sass_store_counts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(stores[0]));w.writeheader();w.writerows(stores)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(dtype,dim) in zip(axes.flat,[(dt,d) for dt in ('float16','bfloat16') for d in (512,1024)]):
        group=[r for r in data if r['dtype']==dtype and int(r['head_dim'])==dim and r['tokens']=='131072' and r['backend'].endswith('/fused_warp')]
        group.sort(key=lambda r:r['backend'])
        median=[float(r['median_us']) for r in group]
        ax.bar([r['backend'].split('/')[0] for r in group],median,
            yerr=[[m-float(r['min_us']) for m,r in zip(median,group)],
                  [float(r['max_us'])-m for m,r in zip(median,group)]],capsize=3)
        ax.set(title=f'{dtype}, d={dim}, fused INT4',ylabel='Graph amortized time (us)')
        ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.2)
    fig.suptitle('H200, 131072 tokens, normalized; medians and min/max of 5 rounds')
    fig.tight_layout();fig.savefig(args.directory/'warp_candidates.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
