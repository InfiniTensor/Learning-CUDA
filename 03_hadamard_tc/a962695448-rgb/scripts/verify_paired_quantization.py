#!/usr/bin/env python3
"""Verify packed pairs with all finite component encodings and offset storage."""
import argparse
import json
from pathlib import Path

from build_torch_extension import load_extension
from verify_execution_context import compare, reference


def guarded_outputs(torch, x):
    shapes = [(*x.shape[:-1], x.shape[-1]//2), tuple(x.shape[:-1])]
    buffers, guards = [], []
    for shape, dtype in zip(shapes, (torch.uint8, torch.float32)):
        size = 1
        for extent in shape: size *= extent
        base = torch.full((size+17,),37,dtype=dtype,device=x.device)
        buffers.append(base[9:-8].reshape(shape));guards.append(base)
    return tuple(buffers), guards


def verify(torch, np, candidate, control=None):
    records=[]
    raw=np.arange(65536,dtype=np.uint16).view(np.int16).copy()
    for dtype in (torch.float16,torch.bfloat16):
        values=torch.from_numpy(raw).view(dtype)
        values=values[torch.isfinite(values.float())]
        assert values.numel()==(63488 if dtype==torch.float16 else 65280)
        anchors=torch.tensor([0,.5,1.5,7,-7,torch.finfo(dtype).tiny,torch.finfo(dtype).max],dtype=dtype)
        for dim in (2,4,8,16):
            for position in (0,dim-1):
                cpu=anchors[torch.arange(values.numel())%anchors.numel()][:,None].expand(-1,dim).clone()
                cpu[:,position]=values
                expected=reference(torch,np,cpu,{'shape':cpu.shape,'method':'quantize_int4'})
                for offset in (0,1):
                    pool=torch.full((cpu.numel()+offset+1,),37,dtype=dtype,device='cuda')
                    x=pool[offset:offset+cpu.numel()].reshape(cpu.shape);x.copy_(cpu)
                    assert x.data_ptr()%4==offset*2
                    for threads in (128,256):
                        compare(torch,candidate.quantize_int4_packed(x,threads),expected)
                        compare(torch,candidate.quantize_int4(x,threads),expected)
                        if control is not None:compare(torch,control.quantize_int4_packed(x,threads),expected)
                        buffers,guards=guarded_outputs(torch,x);pointers=[b.data_ptr() for b in buffers];versions=[b._version for b in buffers]
                        assert candidate.quantize_int4_packed_out(x,*buffers,threads) is None
                        compare(torch,buffers,expected);compare(torch,x,(cpu,))
                        assert [b.data_ptr() for b in buffers]==pointers
                        assert [b._version for b in buffers]==[v+1 for v in versions]
                        for guard in guards:
                            assert torch.all(guard[:9]==37) and torch.all(guard[-8:]==37)
                        if offset:assert pool[0].item()==37
                        assert pool[-1].item()==37
                        records.append({'dtype':str(dtype),'dim':dim,'variable_position':position,'input_offset_elements':offset,'threads':threads,
                            'finite_component_encodings':values.numel(),'cpu_oracle':'PASS','allocating_out_and_original':'PASS','guards_versions_and_input':'PASS'})
    return {'status':'PASS','unique_finite_component_encodings':128768,'records':records,
            'scope':'Each finite encoding is inserted into selected low/high components; other values use seven fixed anchor classes. This is not every vector combination.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import numpy as np
    op=load_extension(build_directory=args.build_directory)
    report=verify(torch,np,op)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS:',len(report['records']),'configurations')


if __name__=='__main__':main()
