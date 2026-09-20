"""Collect MUPTI kernel activities and retain unavailable-timestamp evidence.

An exit code of zero does not mean timestamps are valid: some containers expose
launch records but no mt-perf hardware events. Validate each timestamp explicitly.
Run this separately from benchmark.py so instrumentation never affects timing.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == '03_hadamard_tc'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, default=ROOT / ('build/musa/hadamard' if HADAMARD else 'build/musa/quantize'))
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/musa/profile')
    p.add_argument('--musa-path', type=Path, default=Path('/usr/local/musa'))
    a = p.parse_args()
    binary = a.binary.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lib = ROOT / 'build/musa/libmusa_profile.so'
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', str(ROOT/'tests/musa_profile.cpp'),
               '-I'+str(a.musa_path/'include'), '-L'+str(a.musa_path/'lib'), '-lmupti', '-lmusart', '-pthread', '-o', str(lib)]
    subprocess.run(command, check=True)
    report = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'collector_build': command, 'tool': 'MUPTI activity API', 'runs': []}
    with tempfile.TemporaryDirectory(prefix='musa-profile-') as tmp:
        t = Path(tmp)
        for dtype in ([1, 2] if HADAMARD else [0, 1]):
            rows = 1024
            write_tensor(t/'input', np.random.default_rng(42).normal(size=(rows,1024)).astype(np.float32), dtype)
            for fmt in ['mxfp8', 'nvfp4']:
                name = f'{fmt}_{["fp32","fp16","bf16"][dtype]}'
                trace = out/(name+'.tsv')
                metrics = out/(name+'.json')
                cmd = [str(binary), '--input', str(t/'input'), '--output', str(t/'output'),
                       '--packed', str(t/'packed'), '--log', str(metrics), '--repeats', '3']
                if HADAMARD:
                    cmd += ['--format', fmt, '--materialized_compare', '1']
                else:
                    (t/'config').write_text(f'format = "{fmt}"\nscale_mode = "block"\noutput_type = "fp32"\nrounding = "nearest"\n')
                    cmd += ['--config', str(t/'config')]
                env = dict(os.environ, MUSA_VISIBLE_DEVICES='0', LP_MUPTI_OUTPUT=str(trace),
                           LD_PRELOAD=str(lib), LD_LIBRARY_PATH=str(a.musa_path/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
                (out/(name+'.log')).write_text(proc.stdout+proc.stderr)
                lines = trace.read_text().splitlines() if trace.exists() else []
                records = list(csv.DictReader((line for line in lines if not line.startswith('#')), delimiter='\t'))
                dropped = sum(int(line.split('=')[1]) for line in lines if line.startswith('# dropped_records='))
                valid = [r for r in records if int(r['start_ns'])>0 and int(r['end_ns'])>int(r['start_ns'])]
                groups = collections.defaultdict(list)
                for r in valid:
                    groups[r['kernel']].append((int(r['end_ns'])-int(r['start_ns']))/1000)
                status = ('ERROR' if proc.returncode else 'NO_RECORDS' if not records else
                          'UNAVAILABLE_TIMESTAMPS' if len(valid)!=len(records) else
                          'DROPPED_RECORDS' if dropped else 'PASS')
                report['runs'].append({'case':name,'command':cmd,'exit_code':proc.returncode,'status':status,
                    'records':len(records),'valid_timestamps':len(valid),'dropped_records':dropped,
                    'kernels':[{'name':k,'count':len(v),'mean_us':sum(v)/len(v)} for k,v in groups.items()]})
                (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
                print(name,status,len(records),len(valid),flush=True)
    # Missing instrumentation is a reported capability limit, never a passing check.
    if any(r['status']=='ERROR' for r in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
