#!/usr/bin/env python3
"""Check paired Hadamard storage, rounded transforms and fused INT4 bytes."""
import argparse
import json
import math
from pathlib import Path

from build_torch_extension import load_extension
from verify_execution_context import check_case, compare, reference


def transform_reference(torch, np, cpu, scale):
    values = cpu.float().numpy().copy()
    size = values.shape[-1]
    stride = 1
    while stride < size:
        blocks = values.reshape(-1, size // (2 * stride), 2 * stride)
        left = blocks[..., :stride].copy()
        right = blocks[..., stride:].copy()
        blocks[..., :stride] = left + right
        blocks[..., stride:] = left - right
        stride *= 2
    rounded = torch.from_numpy(values * np.float32(scale)).to(cpu.dtype)
    assert torch.all(torch.isfinite(rounded.float()))
    return rounded


def guarded_outputs(torch, x, method):
    specs = [(tuple(x.shape), x.dtype)] if method == 'hadamard' else [
        ((*x.shape[:-1], x.shape[-1] // 2), torch.uint8),
        (tuple(x.shape[:-1]), torch.float32),
    ]
    buffers, guards = [], []
    for shape, dtype in specs:
        base = torch.full((math.prod(shape) + 17,), 37, dtype=dtype, device=x.device)
        buffers.append(base[9:-8].reshape(shape))
        guards.append(base)
    return tuple(buffers), guards


def verify(torch, np, candidate, control=None):
    records = []
    raw = np.arange(65536, dtype=np.uint16).view(np.int16).copy()
    for dtype in (torch.float16, torch.bfloat16):
        values = torch.from_numpy(raw).view(dtype)
        values = values[torch.isfinite(values.float())]
        assert values.numel() == (63488 if dtype == torch.float16 else 65280)
        # Small anchors keep every transform finite, even for the largest input.
        anchors = torch.tensor([0, .5, -.5, .25, -.25, torch.finfo(dtype).tiny], dtype=dtype)
        for dim in (2, 4, 8, 16):
            for position in (0, dim - 1):
                cpu = anchors[torch.arange(values.numel()) % anchors.numel()][:, None].expand(-1, dim).clone()
                cpu[:, position] = values
                for scale in (1., 1 / math.sqrt(dim)):
                    transformed = transform_reference(torch, np, cpu, scale)
                    expected = {
                        'hadamard': (transformed,),
                        'hadamard_int4': reference(torch, np, transformed, {'method': 'quantize_int4'}),
                    }
                    for offset in (0, 1):
                        pool = torch.full((cpu.numel() + offset + 1,), 37, dtype=dtype, device='cuda')
                        x = pool[offset:offset + cpu.numel()].reshape(cpu.shape)
                        x.copy_(cpu)
                        version = x._version
                        assert x.data_ptr() % 4 == offset * 2
                        for threads in (128, 256):
                            options = dict(scale=scale, block_threads=threads, row_layout='packed')
                            for method in ('hadamard', 'hadamard_int4'):
                                if control is not None:
                                    compare(torch, getattr(control, method)(x, **options), expected[method])
                                compare(torch, getattr(candidate, method)(x, **options), expected[method])
                                buffers, guards = guarded_outputs(torch, x, method)
                                pointers = [b.data_ptr() for b in buffers]
                                versions = [b._version for b in buffers]
                                assert getattr(candidate, method + '_out')(x, *buffers, **options) is None
                                compare(torch, buffers, expected[method])
                                assert [b.data_ptr() for b in buffers] == pointers
                                assert [b._version for b in buffers] == [v + 1 for v in versions]
                                for guard in guards:
                                    assert torch.all(guard[:9] == 37) and torch.all(guard[-8:] == 37)
                            compare(torch, x, (cpu,))
                            assert x._version == version
                            if offset:
                                assert pool[0].item() == 37
                            assert pool[-1].item() == 37
                            records.append({'dtype': str(dtype), 'dim': dim, 'position': position,
                                'scale': scale, 'offset': offset, 'threads': threads,
                                'finite_component_encodings': values.numel(),
                                'both_methods_allocating_and_out': 'PASS', 'guards_versions_and_input': 'PASS'})
    return {'status': 'PASS', 'records': records, 'unique_finite_component_encodings': 128768,
            'scope': 'Selected first/last components contain all finite storage encodings; remaining components use six bounded anchor classes. Not all vector combinations.'}


def contexts(torch, np, op):
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for shape in ((17, 2), (4097, 16), (1, 3, 7, 8)):
            for method in ('hadamard', 'hadamard_int4'):
                for threads in (128, 256):
                    case = dict(shape=shape, method=method, block_threads=threads, row_layout='packed')
                    records.append(check_case(torch, np, op, case, dtype))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import numpy as np
    op = load_extension(build_directory=args.build_directory)
    report = verify(torch, np, op)
    report['contexts'] = contexts(torch, np, op)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', len(report['records']), len(report['contexts']))


if __name__ == '__main__':
    main()
