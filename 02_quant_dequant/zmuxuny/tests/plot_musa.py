"""Standalone plots of the measured S4000 pipeline comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
HADAMARD=ROOT.parent.name=='03_hadamard_tc'
R=ROOT/'results/musa'
d=json.loads((R/'comparison.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
colors=['#8793A4','#2563A6','#31A182']
if not HADAMARD:
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,fmt in zip(axes,['mxfp8','nvfp4']):
        rs=[r for r in d['records'] if r['rows']*r['cols']*(4 if r['dtype']=='fp32' else 2)==128*1024*1024 and r['format']==fmt]
        x=np.arange(len(rs))
        for k,label in enumerate(['before','after']):
            v=[r['median'][label]['quant_ms']*1000 for r in rs]
            bars=ax.bar(x+(k-.5)*.34,v,.34,label=['Initial port','Tuned'][k],color=colors[k])
            ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
        ax.set_xticks(x,[r['dtype'].upper() for r in rs]);ax.set_title(fmt.upper()+' full quantization')
        ax.set_ylabel('Time (microseconds)');ax.set_ylim(0,ax.get_ylim()[1]*1.16);ax.legend(frameon=False)
    fig.suptitle('MTT S4000 | 128 MiB input | median of 3 interleaved trials')
else:
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    rs=[r for r in d['records'] if r['rows']==65536 and r['format']=='mxfp8']
    x=np.arange(len(rs))
    for k,(label,key,title) in enumerate([('before','hadamard_ms','Initial butterfly'),('after','hadamard_ms','Current butterfly'),('after','factorized_tc_ms','Native matrix + butterfly')]):
        bars=ax.bar(x+(k-1)*.25,[r['median'][label][key]*1000 for r in rs],.25,color=colors[k],label=title)
        ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
    ax.set_xticks(x,[r['dtype'].upper() for r in rs]);ax.set_ylabel('Time (microseconds)')
    ax.set_title('MTT S4000 | 128 MiB standalone Hadamard (D=1024)')
    ax.set_ylim(0,ax.get_ylim()[1]*1.25);ax.legend(frameon=False,loc='upper left')
fig.savefig(R/'performance.png')
