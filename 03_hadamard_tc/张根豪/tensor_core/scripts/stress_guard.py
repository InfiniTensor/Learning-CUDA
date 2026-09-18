#!/usr/bin/env python3
"""Amplitude/cancellation sweep for the guarded experimental PTX path."""
import argparse
import csv
import math
from pathlib import Path
from benchmark_library import Bridge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    import torch
    from fast_hadamard_transform import hadamard_transform
    torch.set_num_threads(2);torch.manual_seed(20260911)
    bridge=Bridge(args.library);rows=[]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for dtype in (torch.float16,torch.bfloat16):
        powers=(-16,-8,0,3,5,7) if dtype==torch.float16 else (-120,-64,-16,-8,0,3,5,7,32,100)
        for dim in (64,128,256):
            for norm in (False,True):
                for power in powers:
                    for pattern in ('random','cancellation'):
                        x=torch.randn(512,dim)*2.**power
                        if pattern=='cancellation': x[:,1::2]=-x[:,::2]
                        x=x.to(device='cuda',dtype=dtype)
                        ref=hadamard_transform(x,1/math.sqrt(dim) if norm else 1)
                        y=bridge.operation(x,norm,9)()
                        error=(y.float()-ref.float()).abs().max().item()
                        passed=math.isfinite(error) and error < (.01 if dtype==torch.float16 else .05)
                        rows.append(dict(dtype=str(dtype),dim=dim,normalize=norm,power=power,
                            pattern=pattern,tokens=512,max_abs=error,passed=passed))
    with (args.output_dir/'stress.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'passed={sum(r["passed"] for r in rows)}/{len(rows)}')
    if not all(r['passed'] for r in rows): raise SystemExit(1)


if __name__=='__main__': main()
