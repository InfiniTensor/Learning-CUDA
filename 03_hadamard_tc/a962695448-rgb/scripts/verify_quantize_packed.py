#!/usr/bin/env python3
"""Independent byte-level checks for explicit packed quantization."""
import argparse
import json
import math
from pathlib import Path

from build_torch_extension import load_extension
from verify_execution_context import compare, fixture, reference, require_rejection
from verify_out_buffers import buffers_for


def pattern(torch, shape, dtype, name):
    index = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    if name == 'zero':
        return torch.zeros(shape, dtype=dtype)
    if name == 'finite':
        return fixture(torch, shape, dtype, 1)
    if name == 'ties':
        table = torch.tensor([-7, -6.5, -2.5, -.5, .5, 1.5, 6.5, 7], dtype=dtype)
    elif name == 'extreme':
        maximum = torch.finfo(dtype).max
        table = torch.tensor([maximum, -maximum, maximum/2, -maximum/2, 0, 1, -1, 0], dtype=dtype)
    else:
        smallest = torch.finfo(dtype).tiny / (1024 if dtype == torch.float16 else 128)
        table = torch.tensor([0, smallest, -smallest, smallest*3, -smallest*3, smallest*7, -smallest*7, 0], dtype=dtype)
    values = table[index % table.numel()]
    if name == 'ties' and shape[-1] > 1:
        values[..., -1] = 7  # Exact scale=1 exposes round-to-nearest-even ties.
    return values


def call(op, x, buffers, threads):
    if buffers is None:
        return op.quantize_int4_packed(x, threads)
    assert op.quantize_int4_packed_out(x, *buffers, threads) is None
    return buffers


def verify(torch, np, op, control=None):
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (1, 2, 4, 8, 16):
            for shape in [*( (n, dim) for n in (1, 3, 17, 255, 257, 4097)), (1, 3, 7, dim)]:
                for name in ('zero', 'finite', 'ties', 'extreme', 'subnormal'):
                    cpu = pattern(torch, shape, dtype, name)
                    expected = reference(torch, np, cpu, {'method':'quantize_int4','shape':shape})
                    x = cpu.cuda()
                    for threads in (128, 256):
                        if control is not None:
                            compare(torch, control.quantize_int4(x,threads), expected)
                        compare(torch, op.quantize_int4(x,threads), expected)
                        compare(torch, call(op,x,None,threads), expected)
                        buffers,guards = buffers_for(torch,x,'quantize_int4',True)
                        versions = [b._version for b in buffers]
                        pointers = [b.data_ptr() for b in buffers]
                        compare(torch,call(op,x,buffers,threads),expected)
                        assert [b._version for b in buffers] == [v+1 for v in versions]
                        assert [b.data_ptr() for b in buffers] == pointers
                        for guard in guards:
                            assert torch.all(guard[:8] == 37) and torch.all(guard[-8:] == 37)
                        compare(torch,x,(cpu,))
                        records.append({'dtype':str(dtype),'shape':shape,'pattern':name,'threads':threads,'allocating_and_out_passed':True,'guards_and_versions_passed':True})
    negatives = []
    for label,dim,options in [('large32',32,{}),('large256',256,{}),('bad_threads',8,{'block_threads':64}),('auto_unsupported',8,{'row_layout':'auto'})]:
        x = torch.ones((17,dim),device='cuda',dtype=torch.float16)
        buffers,_ = buffers_for(torch,x,'quantize_int4')
        for b in buffers:b.fill_(3)
        protected=(x,*buffers);before=[v.cpu().clone() for v in protected];versions=[v._version for v in protected]
        for method in ('quantize_int4_packed','quantize_int4_packed_out'):
            args=(x,*buffers) if method.endswith('_out') else (x,)
            try:getattr(op,method)(*args,**options)
            except (ValueError,RuntimeError,TypeError):pass
            else:raise AssertionError('invalid call accepted: '+label)
            for value,original,version in zip(protected,before,versions):
                compare(torch,value,(original,));assert value._version==version
        negatives.append({'case':label,'allocating_and_out_rejected_before_mutation':True})
    return {'status':'PASS','positive':records,'negative':negatives}


def contexts(torch,np,op):
    records=[]
    for dtype in (torch.float16,torch.bfloat16):
        for shape in ((17,1),(4097,16)):
            for threads in (128,256):
                for out_api in (False,True):
                    cpu=[fixture(torch,shape,dtype,phase) for phase in range(4)]
                    expected=[reference(torch,np,c,{'method':'quantize_int4','shape':shape}) for c in cpu]
                    x=torch.zeros_like(cpu[0],device='cuda');payloads=[c.cuda() for c in cpu]
                    buffers=buffers_for(torch,x,'quantize_int4')[0] if out_api else None
                    side=torch.cuda.Stream()
                    for wrong in (False,True):
                        x.zero_();side.wait_stream(torch.cuda.current_stream())
                        for stream in (torch.cuda.default_stream(),side):
                            with torch.cuda.stream(stream):warm=call(op,x,buffers,threads)
                        torch.cuda.synchronize()
                        with torch.cuda.stream(side):
                            torch.cuda._sleep(50_000_000);x.copy_(payloads[1])
                            if wrong:
                                with torch.cuda.stream(torch.cuda.default_stream()):actual=call(op,x,buffers,threads)
                            else:actual=call(op,x,buffers,threads)
                        torch.cuda.synchronize()
                        if wrong:require_rejection(torch,actual,expected[1],'wrong_stream')
                        else:compare(torch,actual,expected[1])
                    side.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=side):actual=call(op,x,buffers,threads)
                    stale=None
                    for phase in range(4):
                        with torch.cuda.stream(side):x.copy_(payloads[phase]);graph.replay()
                        side.synchronize();compare(torch,actual,expected[phase]);compare(torch,x,(cpu[phase],))
                        if phase==0:stale=tuple(v.cpu().clone() for v in actual)
                        if phase==1:require_rejection(torch,stale,expected[phase],'stale_output')
                    records.append({'dtype':str(dtype),'shape':shape,'threads':threads,'out_api':out_api,'replays':4,'negative_controls':2})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    op=load_extension(build_directory=args.build_directory)
    report=verify(torch,np,op);report['contexts']=contexts(torch,np,op)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['positive']),len(report['contexts']))


if __name__=='__main__':main()
