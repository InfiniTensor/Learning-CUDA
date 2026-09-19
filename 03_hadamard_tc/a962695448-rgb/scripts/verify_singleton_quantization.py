#!/usr/bin/env python3
"""Exhaust singleton quantization storage encodings without changing its domain."""
import argparse
import json
from pathlib import Path

from build_torch_extension import load_extension
from verify_execution_context import compare, reference
from verify_out_buffers import buffers_for


def verify(torch, np, candidate, control=None):
    records = []
    words = np.arange(65536, dtype=np.uint16)
    # Fixed permutation places different signs/exponents in adjacent warp lanes.
    np.random.default_rng(9172030).shuffle(words)
    for dtype in (torch.float16, torch.bfloat16):
        expected_finite = 63488 if dtype == torch.float16 else 65280
        for shape in ((65536, 1), (1, 1, 65539, 1)):
            raw = np.resize(words, int(np.prod(shape))).view(np.int16).copy()
            cpu = torch.from_numpy(raw).view(dtype).reshape(shape)
            mask = np.isfinite(cpu.float().numpy().reshape(-1))
            if len(mask) == 65536:
                assert int(mask.sum()) == expected_finite
            finite_input = cpu.reshape(-1, 1)[torch.from_numpy(mask)]
            oracle = reference(torch, np, finite_input, {'shape':finite_input.shape, 'method':'quantize_int4'})
            x = cpu.cuda()
            for threads in (128, 256):
                # Compare nonfinite encodings only with the same legacy packed
                # path; other layouts may choose a different unsupported fallback.
                baseline_op = control if control is not None else candidate
                baseline = tuple(v.cpu() for v in baseline_op.quantize_int4_packed(x, threads))
                actual = candidate.quantize_int4_packed(x, threads)
                compare(torch, actual, baseline)
                actual_finite = tuple(v.cpu().reshape(-1)[torch.from_numpy(mask)] for v in actual)
                expected_finite_outputs = tuple(v.reshape(-1) for v in oracle)
                compare(torch, actual_finite, expected_finite_outputs)
                buffers, guards = buffers_for(torch, x, 'quantize_int4', True)
                versions = [b._version for b in buffers]
                pointers = [b.data_ptr() for b in buffers]
                assert candidate.quantize_int4_packed_out(x, *buffers, threads) is None
                compare(torch, buffers, baseline)
                assert [b._version for b in buffers] == [v+1 for v in versions]
                assert [b.data_ptr() for b in buffers] == pointers
                for guard in guards:
                    assert torch.all(guard[:8] == 37) and torch.all(guard[-8:] == 37)
                compare(torch, x, (cpu,))
                assert torch.all((buffers[0] & 0xF0) == 0)
                records.append({'dtype':str(dtype),'shape':shape,'threads':threads,
                    'storage_encodings':65536,'finite_rows_checked':int(mask.sum()),
                    'independent_finite_oracle':'PASS',
                    'legacy_fallback_bits':'PASS' if control is not None else 'NOT_CHECKED: control module required',
                    'out_guards_versions_and_storage':'PASS','input_unchanged':True})
    return {'status':'PASS','unique_storage_encodings':131072,'unique_finite_values':128768,'records':records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    import torch
    import numpy as np
    module = load_extension(build_directory=args.build_directory)
    report = verify(torch,np,module)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS:',report['unique_finite_values'],'finite values,',report['unique_storage_encodings'],'storage encodings')


if __name__=='__main__':main()
