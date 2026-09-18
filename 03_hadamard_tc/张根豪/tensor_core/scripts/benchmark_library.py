#!/usr/bin/env python3
"""Paired resident-input CUDA Graph throughput vs unmodified Dao v1.1.0.

Graph capture removes Python/ctypes and allocation costs from timed replay on
both sides. Results are graph-replay amortized device time, NOT end-to-end API
latency or independently profiled isolated kernel duration. No speed claims
across devices, scales, dtypes, or quantization protocols.
"""
import argparse
import csv
import ctypes
import json
import math
from pathlib import Path


class Bridge:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(Path(path).resolve()))
        self.run = self.lib.hadamard_run
        self.run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.run.restype = ctypes.c_int

    def operation(self, x, normalize, mode):
        import torch
        tokens, dim = x.shape
        dt = 0 if x.dtype == torch.float16 else 1
        def call():
            if mode in (0, 1, 2, 6, 7, 8, 9):
                y, scale = torch.empty_like(x), None
            else:
                y = torch.empty((tokens, dim // 2), device=x.device, dtype=torch.uint8)
                scale = torch.empty(tokens, device=x.device, dtype=torch.float32)
            rc = self.run(x.data_ptr(), y.data_ptr(), scale.data_ptr() if scale is not None else None,
                          tokens, dim, dt, int(normalize), mode, torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(f'bridge returned {rc}')
            return y if scale is None else (y, scale)
        return call


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', action='append', required=True, help='label=shared_library')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tokens', default='1,128,16384,131072')
    p.add_argument('--dims', default='64,128,256,512,1024')
    p.add_argument('--norms', default='0,1')
    p.add_argument('--backends', default='baseline,optimized,warp,fused_opt,fused_warp')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--nodes', type=int, default=16)
    p.add_argument('--replays', type=int, default=10)
    args = p.parse_args()
    if min(args.rounds, args.nodes, args.replays) < 1: p.error('counts must be positive')
    import torch
    from fast_hadamard_transform import hadamard_transform
    args.output_dir.mkdir(parents=True, exist_ok=False)
    variants = {label: Bridge(path) for label, path in (s.split('=', 1) for s in args.variant)}
    metadata = dict(vars(args), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    cuda=torch.version.cuda, reference='Dao v1.1.0 unmodified',
                    measurement='CUDA Graph replay: resident same input, amortized device time',
                    allocation='output allocations during capture, excluded from timed replay',
                    quant_reference='Dao transform + this project standalone INT4 quantizer',
                    seed=20260909, order='rotate and reverse backend order each round')
    (args.output_dir/'environment.json').write_text(json.dumps(metadata, default=str, indent=2))
    fields = ['gpu','dtype','tokens','head_dim','normalize','backend','round','us',
              'max_abs','pass','nodes','replays','measurement']
    failures = []
    with (args.output_dir/'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for dtype in (torch.float16, torch.bfloat16):
          for dim in map(int, args.dims.split(',')):
           for tokens in map(int, args.tokens.split(',')):
            for norm in map(int, args.norms.split(',')):
                torch.manual_seed(20260909 + dim + tokens)
                x = torch.randn(tokens, dim, device='cuda', dtype=torch.float32).to(dtype)
                scale = 1 / math.sqrt(dim) if norm else 1.0
                reference = hadamard_transform(x, scale)
                quantizer = next(iter(variants.values()))
                qref = quantizer.operation(reference, False, 3)()
                operations = {'dao': lambda: hadamard_transform(x, scale)}
                def dao_quant():
                    temp = hadamard_transform(x, scale)
                    return quantizer.operation(temp, False, 3)()
                operations['dao_plus_quant'] = dao_quant
                for label, bridge in variants.items():
                    for name, mode in [('baseline',0),('optimized',1),('warp',2),('fused_opt',4),('fused_warp',5),
                                       ('wmma_fast',6),('wmma_split',7),('mma_fast',8),('mma_split',9)]:
                        if name not in args.backends.split(','): continue
                        operations[f'{label}/{name}'] = bridge.operation(x, norm, mode)
                graphs, checks, alive = {}, {}, []
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for name, fn in operations.items():
                        actual = fn()
                        if isinstance(actual, tuple):
                            passed = all(torch.equal(a, b) for a, b in zip(actual, qref)); error = ''
                        else:
                            error = (actual.float()-reference.float()).abs().max().item()
                            passed = math.isfinite(error) and error < (0.01 if dtype == torch.float16 else 0.05)
                        checks[name] = (error, passed)
                        if not passed: failures.append([str(dtype),dim,tokens,norm,name,error])
                        for _ in range(10): fn()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=side):
                            for _ in range(args.nodes): result = fn()
                        # Keep final graph outputs alive; intermediate allocations are
                        # handled by each graph's private memory pool.
                        alive.append(result); graphs[name] = graph
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                names = list(graphs)
                for repeat in range(args.rounds):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    if repeat % 2: order.reverse()
                    for name in order:
                        graph = graphs[name]
                        for _ in range(3): graph.replay()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.replays): graph.replay()
                        end.record(); end.synchronize()
                        error, passed = checks[name]
                        writer.writerow(dict(gpu=metadata['gpu'], dtype=str(dtype).split('.')[-1], tokens=tokens,
                            head_dim=dim, normalize=norm, backend=name, round=repeat,
                            us=start.elapsed_time(end)*1000/(args.nodes*args.replays), max_abs=error,
                            **{'pass':passed}, nodes=args.nodes, replays=args.replays,
                            measurement='graph_amortized_device'))
                    stream.flush()
                print(str(dtype), dim, tokens, norm, 'checked', len(graphs), flush=True)
                del graphs, alive, graph, result, operations, actual, reference, qref, x
                torch.cuda.empty_cache()
    (args.output_dir/'status.json').write_text(json.dumps(dict(failures=failures, all_pass=not failures), indent=2))
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
