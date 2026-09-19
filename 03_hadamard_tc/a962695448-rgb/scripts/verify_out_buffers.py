#!/usr/bin/env python3
"""Verify reusable output buffers, mutation tracking, and execution contexts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from build_torch_extension import load_extension
from verify_execution_context import compare, fixture, reference, require_rejection

METHODS = ("hadamard", "hadamard_int4", "quantize_int4")


def buffers_for(torch, x, method, guarded=False):
    specs = [(tuple(x.shape), x.dtype)] if method == "hadamard" else [
        ((*x.shape[:-1], (x.shape[-1] + 1) // 2), torch.uint8),
        (tuple(x.shape[:-1]), torch.float32),
    ]
    buffers, guards = [], []
    for shape, dtype in specs:
        if guarded:
            base = torch.full((math.prod(shape) + 16,), 37, dtype=dtype, device=x.device)
            buffers.append(base[8:-8].reshape(shape))
            guards.append(base)
        else:
            buffers.append(torch.empty(shape, dtype=dtype, device=x.device))
    return tuple(buffers), guards


def invoke(op, x, buffers, method, **options):
    result = getattr(op, method + "_out")(x, *buffers, **options)
    if result is not None:
        raise AssertionError("out interfaces must return None")


def positive_cases(torch, np, op):
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            for shape in ((17, dim), (1, 1, 3, dim)):
                for method in METHODS:
                    options = [{"block_threads": n} for n in (128, 256)]
                    if method != "quantize_int4":
                        options += [{"row_layout": "auto"}]
                        if dim <= 16:
                            options += [{"row_layout": "packed", "block_threads": n} for n in (128, 256)]
                        if method == "hadamard_int4" and dim == 256:
                            options += [{"fused_layout": "contiguous256"}]
                    for kwargs in options:
                        x = torch.empty(shape, dtype=dtype, device="cuda")
                        buffers, guards = buffers_for(torch, x, method, True)
                        pointers = tuple(b.data_ptr() for b in buffers)
                        call_options = dict(kwargs)
                        if method != "quantize_int4":
                            call_options["scale"] = 1 / math.sqrt(dim)
                        for phase in (1, 2):
                            cpu = fixture(torch, shape, dtype, phase)
                            x.copy_(cpu)
                            expected = reference(torch, np, cpu, {"shape": shape, "method": method})
                            versions = tuple(b._version for b in buffers)
                            invoke(op, x, buffers, method, **call_options)
                            compare(torch, buffers, expected)
                            compare(torch, x, (cpu,))
                            if tuple(b.data_ptr() for b in buffers) != pointers:
                                raise AssertionError("output storage was replaced")
                            if tuple(b._version for b in buffers) != tuple(v + 1 for v in versions):
                                raise AssertionError("successful mutation was not tracked")
                            for base in guards:
                                if not torch.all(base[:8] == 37) or not torch.all(base[-8:] == 37):
                                    raise AssertionError("output guard elements were overwritten")
                        records.append({"dtype": str(dtype), "shape": shape, "method": method,
                                        "options": call_options, "two_inputs_passed": True,
                                        "storage_preserved": True, "guards_preserved": True})
    return records


def reject_without_mutation(torch, op, x, buffers, method, options, label):
    protected = (x, *buffers)
    before = tuple(value.detach().clone() for value in protected)
    versions = tuple(None if value.is_inference() else value._version for value in protected)
    try:
        invoke(op, x, buffers, method, **options)
    except (RuntimeError, ValueError) as error:
        reason = str(error).splitlines()[0]
    else:
        raise AssertionError("invalid call accepted: " + label)
    for value, original, version in zip(protected, before, versions):
        compare(torch, value.detach().resolve_conj().resolve_neg(), (original.cpu(),))
        if version is not None and value._version != version:
            raise AssertionError("rejected call changed a version counter: " + label)
    return {"method": method, "case": label, "rejected_before_mutation": True, "error": reason}


def negative_cases(torch, op):
    records = []
    for method in METHODS:
        labels = ["cpu_input", "noncontiguous_input", "input_grad", "input_lazy_negative",
                  "cpu_output", "wrong_output_dtype", "wrong_output_rank", "wrong_output_shape",
                  "noncontiguous_output", "bad_threads", "input_output_overlap"]
        if method != "quantize_int4":
            labels += ["bad_scale", "bad_layout"]
        if method == "hadamard":
            labels += ["output_grad", "output_lazy_negative", "partial_overlap", "dlpack_overlap"]
        else:
            labels += ["wrong_scales_dtype", "wrong_scales_shape", "wrong_scales_rank", "cpu_scales", "noncontiguous_scales", "scales_grad", "scales_lazy_negative",
                       "input_scales_overlap", "output_output_overlap", "inference_scales"]
        if method == "hadamard_int4": labels += ["bad_fused_layout", "invalid_contiguous256"]
        for label in labels:
            x = torch.arange(192, device="cuda", dtype=torch.float16).reshape(3, 64) / 16
            buffers, _ = buffers_for(torch, x, method)
            buffers = list(buffers)
            for b in buffers:
                b.fill_(3)
            options = {}
            if label == "cpu_input": x = x.cpu()
            elif label == "noncontiguous_input": x = x.T.contiguous().T
            elif label == "input_grad": x.requires_grad_(True)
            elif label == "input_lazy_negative": x = torch._neg_view(x)
            elif label == "cpu_output": buffers[0] = buffers[0].cpu()
            elif label == "wrong_output_dtype": buffers[0] = buffers[0].float() if method != "hadamard" else buffers[0].bfloat16()
            elif label == "wrong_output_rank": buffers[0] = buffers[0].reshape(-1)
            elif label == "wrong_output_shape": buffers[0] = torch.full((3, buffers[0].shape[-1]+1), 3, device="cuda", dtype=buffers[0].dtype)
            elif label == "noncontiguous_output": buffers[0] = buffers[0].T.contiguous().T
            elif label == "bad_threads": options["block_threads"] = 64
            elif label == "bad_scale": options["scale"] = float("nan")
            elif label == "bad_layout": options["row_layout"] = "invalid"
            elif label == "output_grad": buffers[0].requires_grad_(True)
            elif label == "output_lazy_negative": buffers[0] = torch._neg_view(buffers[0])
            elif label == "input_output_overlap": buffers[0] = x if method == "hadamard" else x.view(torch.uint8).reshape(-1)[:96].reshape(3,32)
            elif label == "partial_overlap":
                base = torch.arange(193, device="cuda", dtype=torch.float16)
                x, buffers[0] = base[:192].reshape(3,64), base[1:].reshape(3,64)
            elif label == "dlpack_overlap":
                buffers[0] = torch.from_dlpack(x)
                if buffers[0].data_ptr() != x.data_ptr(): raise AssertionError("DLPack fixture did not alias")
            elif label == "wrong_scales_dtype": buffers[1] = buffers[1].half()
            elif label == "wrong_scales_shape": buffers[1] = torch.ones(4, device="cuda")
            elif label == "wrong_scales_rank": buffers[1] = buffers[1].reshape(3,1)
            elif label == "cpu_scales": buffers[1] = buffers[1].cpu()
            elif label == "noncontiguous_scales": buffers[1] = torch.ones(6,device="cuda")[::2]
            elif label == "scales_grad": buffers[1].requires_grad_(True)
            elif label == "scales_lazy_negative": buffers[1] = torch._neg_view(buffers[1])
            elif label == "input_scales_overlap": buffers[1] = x.view(torch.float32).reshape(-1)[:3]
            elif label == "output_output_overlap":
                pool = torch.full((96,), 3, device="cuda", dtype=torch.uint8)
                buffers[0], buffers[1] = pool.reshape(3,32), pool[:12].view(torch.float32)
            elif label == "inference_scales":
                with torch.inference_mode(): buffers[1] = torch.ones(3, device="cuda")
            elif label == "bad_fused_layout": options["fused_layout"] = "invalid"
            elif label == "invalid_contiguous256": options["fused_layout"] = "contiguous256"
            records.append(reject_without_mutation(torch, op, x, buffers, method, options, label))
    return records


def mutation_contract(torch, np, op):
    records = []
    for method in METHODS:
        x = torch.ones((3,64), device="cuda", dtype=torch.float16)
        buffers, _ = buffers_for(torch, x, method)
        watched = buffers[0] if method == "hadamard" else buffers[1]
        watched.fill_(1)
        grad_input = torch.ones_like(watched, requires_grad=True)
        loss = (grad_input * watched).sum()
        invoke(op, x, buffers, method)
        try: loss.backward()
        except RuntimeError as error:
            if "modified by an inplace operation" not in str(error): raise
        else: raise AssertionError("autograd failed to detect output mutation")
        with torch.inference_mode():
            inference_buffers, _ = buffers_for(torch, x, method)
            invoke(op, x, inference_buffers, method)
        transformed = torch.zeros((3,64),dtype=torch.float16); transformed[:,0] = 64
        expected = (transformed,) if method == "hadamard" else reference(
            torch,np,transformed if method == "hadamard_int4" else x.cpu(),
            {"shape":x.shape,"method":"quantize_int4"})
        compare(torch,inference_buffers,expected)
        # Allocation check starts after warmup, excluding caller-owned buffers.
        for _ in range(10): invoke(op, x, buffers, method)
        torch.cuda.synchronize(); live = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        for _ in range(50): invoke(op, x, buffers, method)
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() != live: raise AssertionError("out path allocated tensor memory")
        records.append({"method":method,"autograd_detected_mutation":True,"inference_mode_passed":True,"output_allocation_bytes":0})
    # Disjoint slices of one allocation are legal, even with a shared version counter.
    pool = torch.zeros(800, device="cuda", dtype=torch.uint8)
    x = pool[16:400].view(torch.float16).reshape(3,64)
    out = pool[400:784].view(torch.float16).reshape(3,64)
    x.fill_(1); expected = torch.zeros_like(x).cpu(); expected[:,0] = 64
    invoke(op, x, (out,), "hadamard"); compare(torch, out, (expected,))
    if not torch.all(x == 1): raise AssertionError("disjoint input slice changed")
    records.append({"case":"disjoint_shared_storage","passed":True})
    return records


def execution_contexts(torch, np, op):
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for shape in ((17,8),(17,256),(1,1,3,64),(4097,8)):
            for method in METHODS:
                cpu = [fixture(torch,shape,dtype,phase) for phase in range(4)]
                expected = [reference(torch,np,v,{"shape":shape,"method":method}) for v in cpu]
                x = torch.zeros_like(cpu[0], device="cuda"); payloads = [v.cuda() for v in cpu]
                buffers, _ = buffers_for(torch,x,method)
                options = {} if method == "quantize_int4" else {"scale":1/math.sqrt(shape[-1]),"row_layout":"auto"}
                side = torch.cuda.Stream()
                for wrong in (False, True):
                    x.zero_(); side.wait_stream(torch.cuda.current_stream())
                    for stream in (torch.cuda.default_stream(),side):
                        with torch.cuda.stream(stream): invoke(op,x,buffers,method,**options)
                    torch.cuda.synchronize()
                    with torch.cuda.stream(side):
                        torch.cuda._sleep(50_000_000); x.copy_(payloads[1])
                        if wrong:
                            with torch.cuda.stream(torch.cuda.default_stream()): invoke(op,x,buffers,method,**options)
                        else: invoke(op,x,buffers,method,**options)
                    torch.cuda.synchronize()
                    if wrong: require_rejection(torch,buffers,expected[1],"consumer_on_default_stream")
                    else: compare(torch,buffers,expected[1])
                side.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=side): invoke(op,x,buffers,method,**options)
                stale = None
                for phase in range(4):
                    with torch.cuda.stream(side): x.copy_(payloads[phase]); graph.replay()
                    side.synchronize(); compare(torch,buffers,expected[phase]); compare(torch,x,(cpu[phase],))
                    if phase == 0: stale = tuple(v.cpu().clone() for v in buffers)
                    elif phase == 1: require_rejection(torch,stale,expected[phase],"stale_graph_output")
                records.append({"dtype":str(dtype),"shape":shape,"method":method,"stream_passed":True,"graph_replays":4,"negative_controls_detected":2})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--json',type=Path,required=True)
    args=parser.parse_args();args.json.parent.mkdir(parents=True,exist_ok=True)
    with args.json.open('x') as stream:
        report={'status':'RUNNING','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        try:
            import numpy as np
            import torch
            op=load_extension(verbose=True,build_directory=str(args.build_directory))
            report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                                   'extension_sha256':hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest()}
            report['positive']=positive_cases(torch,np,op)
            report['negative']=negative_cases(torch,op)
            report['mutation']=mutation_contract(torch,np,op)
            report['contexts']=execution_contexts(torch,np,op)
            report['status']='PASS'
        except Exception: report.update(status='FAIL',traceback=traceback.format_exc())
        root=Path(__file__).resolve().parents[1]
        report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['src/torch_binding.cu','scripts/verify_out_buffers.py']}
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'positive':len(report.get('positive',[])),'negative':len(report.get('negative',[])),'contexts':len(report.get('contexts',[]))}))
    if report['status']!='PASS': print(report['traceback']); return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
