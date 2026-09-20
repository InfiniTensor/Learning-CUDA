#!/usr/bin/env python3
"""从 software 或任意目录运行。所有子进程失败立即停止；profiling 限制单独记录。"""
import argparse, csv, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['quick','full'])
p.add_argument('--run-id', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
p.add_argument('--arch', default='86')
a = p.parse_args()
if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_id):
    p.error('run-id must be a simple directory name')
run = ROOT/'results'/a.run_id
run.mkdir(parents=True, exist_ok=False)
(run/'configs').mkdir(); (run/'data').mkdir(); (run/'csv').mkdir()
commands=[]
def call(cmd, name, cwd=ROOT, optional=False):
    cmd=list(map(str,cmd)); commands.append({'argv':cmd,'cwd':str(cwd),'log':name})
    (run/'commands.json').write_text(json.dumps(commands,indent=2))
    (run/name).parent.mkdir(parents=True,exist_ok=True)
    with (run/name).open('w') as log:
        result=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        if optional:
            print('PROFILING UNAVAILABLE:',name,flush=True)
        else:
            print((run/name).read_text()[-12000:],file=sys.stderr)
            raise RuntimeError('failed: '+str(cmd))
    return result.returncode
meta={'mode':a.mode,'run_id':a.run_id,'seed':42,'arch':a.arch,'warmup':3,'repeat':5 if a.mode=='full' else 3,'cpu_threads':1,'cpu_flags':'Release -O3 -ffp-contract=off','cuda_flags':'-O3 --fmad=false -lineinfo -arch=sm_'+a.arch,'distributions':{'uniform':'[-1,1]','normal':'mean=0,std=1','outliers':'normal; every 100th element *=100'},'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['src','include','tests','scripts','configs'] for f in (ROOT/d).glob('*') if f.is_file()}}
(run/'metadata.json').write_text(json.dumps(meta,indent=2))
for src in (ROOT/'configs').glob('*.toml'): shutil.copy2(src,run/'configs'/src.name)
call(['bash','-c','date -Is; uname -a; cat /etc/os-release; lscpu; c++ --version; cmake --version; nvcc --version; nvidia-smi; command -v compute-sanitizer ncu nsys'],'environment.log')
call(['cmake','-S',ROOT,'-B',ROOT/'build','-DCMAKE_BUILD_TYPE=Release','-DCUDA_ARCH='+a.arch], 'configure.log')
call(['cmake','--build',ROOT/'build','-j','4'],'build.log')
shutil.copy2(ROOT/'build/CMakeCache.txt',run/'CMakeCache.txt')
call(['ctest','--output-on-failure','-V'],'tests.log',ROOT/'build')
call(['nvcc','-std=c++17','-O3','-arch=sm_'+a.arch,ROOT/'tests/probe.cu','-o',run/'probe'],'probe-build.log')
call([run/'probe'],'probe.log')
call([ROOT/'build/quality'],'quality.csv')
lp=ROOT/'build/lp'
# 同一输入文件供两个 variant 使用；顺序交替，避免所有 baseline 都在同一温度阶段。
sizes=[(32,32),(512,512)] if a.mode=='quick' else [(32,32),(512,512),(2048,2048)]
case_id=0
for rows,cols in sizes:
 for dt in ['fp32','fp16']:
  distributions=['normal']
  if a.mode=='full' and rows==512: distributions+=['uniform','outliers']
  for dist in distributions:
   tag=f'{rows}x{cols}_{dt}_{dist}'; inp=run/'data'/(tag+'.tensor')
   call([lp,'generate','--rows',rows,'--cols',cols,'--dtype',dt,'--distribution',dist,'--seed',42,'--output',inp],tag+'_generate.log')
   for fmt in ['e4m3','e5m2','nvfp4']:
    cfg=run/'configs'/(fmt+'.toml')
    for variant in (['baseline','optimized'] if case_id%2==0 else ['optimized','baseline']):
     name=tag+'_'+fmt+'_'+variant
     call([lp,'benchmark','--input',inp,'--config',cfg,'--csv',run/'csv'/(name+'.csv'),'--workdir',run/'data'/name,'--kernel_variant',variant,'--warmup',3,'--repeat',meta['repeat']],name+'.log')
     print('measured',name,flush=True)
    case_id+=1
# 实际独立进程端到端命令。
small=run/'data'/'32x32_fp16_normal.tensor'
for fmt in ['e4m3','e5m2','nvfp4']:
 packed=run/'data'/(fmt+'.lp')
 call([lp,'quantize','--input',small,'--config',run/'configs'/(fmt+'.toml'),'--output',packed,'--backend','cuda','--kernel_variant','optimized'],fmt+'_quantize.log')
 call([lp,'dequantize','--input',packed,'--output',run/'data'/(fmt+'_decoded.tensor'),'--backend','cuda'],fmt+'_dequantize.log')
 call([lp,'verify','--input',small,'--packed',packed],fmt+'_verify.log')
if a.mode=='full':
 for tool in ['memcheck','racecheck']:
  call(['compute-sanitizer','--tool',tool,'--error-exitcode','1',ROOT/'build/integration_tests','gpu'],tool+'.log',run)
 # profiler 统一输出到 run/nsys/；用 1024² 输入，一次覆盖量化 + 反量化
 nsys_dir=run/'nsys'; nsys_dir.mkdir(exist_ok=True)
 prof_in=nsys_dir/'input.tensor'
 call([lp,'generate','--rows','1024','--cols','1024','--dtype','fp32','--distribution','normal','--seed','42','--output',prof_in],'nsys/generate.log')
 prof_cmd=[lp,'benchmark','--input',prof_in,'--config',run/'configs'/'nvfp4.toml','--kernel_variant','optimized','--warmup','0','--repeat','1','--workdir',nsys_dir/'profile_work','--csv',nsys_dir/'profile.csv']
 statuses={}
 if shutil.which('ncu'):
  statuses['ncu']=call(['ncu','--set','basic','--launch-count','3','--export',nsys_dir/'ncu']+prof_cmd,'nsys/ncu.log',optional=True)
 if shutil.which('nsys'):
  statuses['nsys']=call(['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--output',nsys_dir/'nsys_all','--force-overwrite=true']+prof_cmd,'nsys/nsys.log',optional=True)
  if statuses['nsys']==0 and (nsys_dir/'nsys_all.nsys-rep').is_file():
   try:
    with (nsys_dir/'nsys-stats.log').open('w') as f:
     subprocess.run(['nsys','stats','--force-export=true','--report','cuda_gpu_kern_sum,cuda_api_sum',str(nsys_dir/'nsys_all.nsys-rep')],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('wrote nsys/nsys-stats.log',flush=True)
   except Exception as e:
    print('nsys stats failed:',e,flush=True)
    statuses['nsys_stats']=1
 (run/'profiling_status.json').write_text(json.dumps(statuses,indent=2))
call(['nvidia-smi'],'gpu-final.log')
call([sys.executable,ROOT/'scripts/summarize.py',run],'summary.log')
if a.mode=='full':
 call([sys.executable,ROOT/'scripts/write_report.py',run],'report.log')
(run/'SUCCESS').write_text('All required commands completed. See profiling_status.json for optional profiler limitations.\n')
print('Completed:',run,flush=True)