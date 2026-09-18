#!/usr/bin/env python3
"""132-case in-process reference-library validation of non-TC tuning variants."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from benchmark_library import Bridge
from validate_tc import cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dims',default='64,128,256,512,1024')
    p.add_argument('--backends',default='warp,fused_warp')
    args=p.parse_args()
    import torch
    from fast_hadamard_transform import hadamard_transform
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    variants={k:Bridge(v) for k,v in (x.split('=',1) for x in args.variant)}
    records=[]; case_count=0
    for index,(dtype,dim,norm,tokens,pattern) in enumerate(cases()):
        if dim not in list(map(int,args.dims.split(','))): continue
        case_count+=1
        torch.manual_seed(20260908+index)
        native=torch.float16 if dtype=='fp16' else torch.bfloat16
        x=torch.randn(tokens,dim)
        if pattern=='zeros': x.zero_()
        elif pattern=='constant': x.fill_(.25)
        elif pattern=='alternating': x[:]=(torch.arange(dim)%2*2-1)*.25
        elif pattern=='impulse': x.zero_(); x[:,0]=1
        elif pattern=='uniform': x.uniform_(-1,1)
        elif pattern=='outlier': x.mul_(.01); x[:,0]=32
        x=x.to(device='cuda',dtype=native)
        ref=hadamard_transform(x,1/math.sqrt(dim) if norm else 1)
        qref=next(iter(variants.values())).operation(ref,False,3)()
        for label,bridge in variants.items():
            for name,mode in [('warp',2),('fused_warp',5),('wmma_fast',6),('wmma_split',7),('mma_fast',8),('mma_split',9)]:
                if name not in args.backends.split(','): continue
                actual=bridge.operation(x,norm,mode)()
                if isinstance(actual,tuple):
                    passed=all(torch.equal(a,b) for a,b in zip(actual,qref)); error=''
                    raw=b''.join(a.cpu().numpy().tobytes() for a in actual)
                else:
                    error=(actual.float()-ref.float()).abs().max().item()
                    passed=math.isfinite(error) and error < (.01 if dtype=='fp16' else .05)
                    raw=actual.view(torch.uint16).cpu().numpy().tobytes()
                records.append(dict(case=index,dtype=dtype,dim=dim,normalize=norm,tokens=tokens,
                    pattern=pattern,variant=label,backend=name,max_abs=error,passed=passed,
                    sha256=hashlib.sha256(raw).hexdigest()))
    with (args.output_dir/'validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    failed=[r for r in records if not r['passed']]
    (args.output_dir/'status.json').write_text(json.dumps(dict(cases=case_count,variants=list(variants),
        rows=len(records),failed=failed,reference='Dao v1.1.0',all_pass=not failed),indent=2))
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
