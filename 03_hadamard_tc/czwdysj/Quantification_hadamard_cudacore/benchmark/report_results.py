#!/usr/bin/env python3
import argparse, csv, math, statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("raw"); p.add_argument("report"); p.add_argument("--summary",required=True); a=p.parse_args()
rows=list(csv.DictReader(open(a.raw,newline="")))
groups=defaultdict(list)
for r in rows: groups[(r["implementation"],int(r["batch"]),int(r["head_dim"]),r["dtype"])].append(r)
summary=[]
for key, rs in sorted(groups.items(),key=lambda x:(x[0][3],x[0][1],x[0][2],x[0][0])):
    lat=[float(r["latency_us"]) for r in rs]; median=statistics.median(lat); mn=min(lat); mx=max(lat); variation=(mx-mn)/median if median else 0
    summary.append(dict(implementation=key[0],batch=key[1],head_dim=key[2],dtype=key[3],latency_us=median,min_us=mn,max_us=mx,
        variation=variation,unstable=variation>0.05,correct=all(r["correct"]=="true" for r in rs),max_quant_error=max(int(r["max_quant_error"]) for r in rs),
        max_amax_error=max(float(r["max_amax_error"]) for r in rs),max_scale_error=max(float(r["max_scale_error"]) for r in rs)))
lookup={(r["batch"],r["head_dim"],r["dtype"],r["implementation"]):r for r in summary}
for r in summary:
    base=lookup.get((r["batch"],r["head_dim"],r["dtype"],"separate")); valid=bool(base and r["correct"] and base["correct"])
    r["speedup_vs_separate"]=base["latency_us"]/r["latency_us"] if valid else ""
    r["latency_ratio_vs_separate"]=r["latency_us"]/base["latency_us"] if valid else ""
    r["percent_latency_reduction"]=(1-r["latency_us"]/base["latency_us"])*100 if valid else ""
fields=list(summary[0]);
with open(a.summary,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
lines=["# Hadamard + rowwise INT8 quantization benchmark", "", "All latency values are three-run medians measured with CUDA events. Positive reduction means the fused kernel is faster than the equivalent three-kernel baseline.", ""]
for dtype in ("fp32","fp16","bf16"):
    lines += [f"## {dtype.upper()}", "", "| Shape | Version | Latency (us) | Speedup vs separate | Reduction | Stable | Correct |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["dtype"]!=dtype: continue
        speed=r["speedup_vs_separate"]; red=r["percent_latency_reduction"]
        lines.append(f'| [{r["batch"]}, {r["head_dim"]}] | {r["implementation"]} | {r["latency_us"]:.3f} | {speed:.3f}x | {red:+.1f}% | {"UNSTABLE" if r["unstable"] else "yes"} | {"PASS" if r["correct"] else "FAIL"} |')
    lines.append("")
lines += ["## Aggregate", "", "Geometric mean speedup is computed over valid representative cases.", "", "| Version | Geomean speedup vs separate | Wins | Cases |", "|---|---:|---:|---:|"]
for version in ("fused_v0","fused_v1","fused_v2"):
    vals=[float(r["speedup_vs_separate"]) for r in summary if r["implementation"]==version and r["speedup_vs_separate"]!=""]
    gm=math.exp(sum(math.log(x) for x in vals)/len(vals)); lines.append(f"| {version} | {gm:.3f}x | {sum(x>1 for x in vals)} | {len(vals)} |")
Path(a.report).write_text("\n".join(lines)+"\n")
