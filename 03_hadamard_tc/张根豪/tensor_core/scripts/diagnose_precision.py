#!/usr/bin/env python3
"""Localize split errors using actual GPU P and pre-cast output, plus FP64 CPU stages."""
import argparse
import csv
import ctypes
import json
from pathlib import Path
from benchmark_library import Bridge


def fht(x):
    y=x.clone()
    d=y.shape[-1]; stride=1
    while stride<d:
        v=y.reshape(-1,d//(2*stride),2,stride)
        a,b=v[:,:,0].clone(),v[:,:,1].clone()
        v[:,:,0],v[:,:,1]=a+b,a-b
        stride*=2
    return y


def finish(p,dim):
    # Per 16x16 tile: left H_m, then cross-tile butterfly for d>256.
    m=min(dim,256)//16
    y=fht(p.reshape(-1,m,16).transpose(1,2).contiguous()).transpose(1,2).contiguous()
    y=y.reshape(-1,dim//256 if dim>256 else 1,256 if dim>256 else dim)
    if dim>256: y=fht(y.transpose(1,2).contiguous()).transpose(1,2).contiguous()
    return y.reshape(-1,dim)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True)
    p.add_argument('--validation-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    bridge=Bridge(args.library)
    trace=bridge.lib.hadamard_tc_trace
    trace.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
    trace.restype=ctypes.c_int
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.validation_dir/'validation.csv').open() as f:
        failed=[r for r in csv.DictReader(f) if r['implementation']=='tc_split' and r['pass']!='True']
    records=[]; examples=[]
    for row in failed:
        name=row['case']; parts=name.split('_'); dim=int(parts[2][1:]); tokens=int(parts[3][1:])
        native=torch.float16 if parts[1]=='fp16' else torch.bfloat16
        x=torch.frombuffer(bytearray((args.validation_dir/'failures'/name/'input.bin').read_bytes()),dtype=native).reshape(tokens,dim)
        ref=torch.frombuffer(bytearray((args.validation_dir/'failures'/name/'ref.bin').read_bytes()),dtype=native).reshape(tokens,dim)
        per=max(1,256//dim); padded=(tokens+per-1)//per*per
        gx=torch.zeros(padded,dim,device='cuda',dtype=native); gx[:tokens]=x.cuda()
        gp=torch.empty_like(gx,dtype=torch.float32); gy=torch.empty_like(gp)
        rc=trace(gx.data_ptr(),gp.data_ptr(),gy.data_ptr(),padded,dim,int(native==torch.bfloat16),torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(rc)
        actual=bridge.operation(gx,False,7)().cpu()
        precast=gy.cpu(); p_gpu=gp.cpu()
        torch.cuda.synchronize()
        if not torch.equal(precast.to(native),actual): raise RuntimeError('trace changes native output')
        # Complete original tiles only: original tails use warp, not TC.
        count=tokens//per*per
        xx=gx.cpu().double()
        p_exact=fht(xx.reshape(-1,16)).reshape(padded,dim)
        pp=p_gpu.double()
        hi=p_gpu.to(native).double()
        lo=(p_gpu-hi.float()).to(native).double()
        exact=fht(xx)
        after_p=finish(pp,dim)
        after_split=finish(hi+lo,dim)
        err=(actual[:count].float()-ref[:count].float()).abs()
        flat=int(err.argmax()); t,c=divmod(flat,dim)
        records.append(dict(case=name,checked_tokens=count,trace_bit_exact=True,
            p_max_abs=float((pp-p_exact).abs().max()),
            split_representation_max_abs=float((hi+lo-pp).abs().max()),
            first_mma_propagated_max_abs=float((after_p-exact).abs().max()),
            split_representation_propagated_max_abs=float((after_split-after_p).abs().max()),
            later_gpu_accum_max_abs=float((precast.double()-after_split).abs().max()),
            native_max_abs=float(err.max()),
            rounded_exact_vs_library_max_abs=float((exact[:count].to(native).float()-ref[:count].float()).abs().max())))
        examples.append(dict(case=name,token=t,channel=c,library=float(ref[t,c]),actual=float(actual[t,c]),
            exact_fp64=float(exact[t,c]),after_first_mma=float(after_p[t,c]),
            after_hi_lo_representation=float(after_split[t,c]),gpu_before_cast=float(precast[t,c]),
            input=[float(v) for v in x[t]]))
    with (args.output_dir/'stages.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (args.output_dir/'worst_elements.json').write_text(json.dumps(examples,indent=2))
    (args.output_dir/'status.json').write_text(json.dumps(dict(cases=len(records),trace_bit_exact=True,
        note='FP64 decomposition is a diagnostic, not a replacement PDF acceptance reference'),indent=2))


if __name__=='__main__': main()
