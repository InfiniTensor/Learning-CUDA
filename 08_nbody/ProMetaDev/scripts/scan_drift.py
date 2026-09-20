
import subprocess, itertools, os, re, sys, pathlib

# 项目根目录 = 本脚本所在目录的上一级（不依赖任何绝对路径）
root = pathlib.Path(__file__).resolve().parent.parent
os.chdir(root)
exec_path = root / 'build' / 'nbody'
particle = root / 'data' / 'particles_4096_plummer.txt'
params   = root / 'data' / 'params_default.txt'
results = []
dts = [1e-4, 5e-4, 1e-3, 2e-3, 3e-3]
epss = [0.005, 0.01, 0.02, 0.05, 0.1]
for dt, eps in itertools.product(dts, epss):
    out_bin = root / 'outputs' / f'_s2.bin'
    out_log = root / 'outputs' / f'_s2.log'
    cmd = [str(exec_path), str(particle), str(params), str(out_bin), str(out_log),
           '--kernel', 'tiling', '--integrator', 'leapfrog', '--no-cpu', '--check-energy',
           '--softening', str(eps), '--dt', str(dt), '--steps', '1000', '--record', '100']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    txt = p.stdout + p.stderr
    m_rel  = re.findall(r'\|\u0394E/E0\|\s*[:=]\s*([+-]?[\d.eE+-]+)', txt)          # unicode Δ
    m_rel2 = re.findall(r'E_drift.*?([\d.eE+-]+)', txt, flags=re.S)                 # fallback
    drift = float(m_rel[-1]) if m_rel else (float(m_rel2[-1]) if m_rel2 else float('nan'))
    drift_pct = drift * 100.0 if drift >= 0 else abs(drift)*100.0
    m_dt   = re.search(r'avg step time\s*:\s*([\d.eE+-]+)\s*ms', txt)
    m_tp   = re.search(r'throughput\s*:\s*([\d.eE+-]+)', txt)
    step_ms = float(m_dt.group(1)) if m_dt else float('nan')
    tp = float(m_tp.group(1)) if m_tp else float('nan')
    ok = drift_pct < 10.0
    print(f'dt={dt:6g} eps={eps:5g}  dE/E0 % = {drift_pct:8.4f}%  step={step_ms:6.3f}ms  tp={tp:10.0f}  ok={ok}')
    results.append((dt,eps,drift_pct,step_ms,tp,ok))
print('---- SORTED by drift ----')
results.sort(key=lambda x:x[2])
for r in results[:10]: print(r)
print('ok count =', sum(1 for r in results if r[5]), '/', len(results))
