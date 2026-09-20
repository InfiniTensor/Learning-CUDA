#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes,csv,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path
import torch
from fast_hadamard_transform import hadamard_transform

ROOT=Path(__file__).resolve().parents[1]
VERSIONS={"fused_v0":0,"fused_v1":1,"fused_v2":2,"separate":3}
DTYPES={"fp32":(torch.float32,0),"fp16":(torch.float16,1),"bf16":(torch.bfloat16,2)}
IMPLS=(*VERSIONS,"pytorch_eager","pytorch_compiled","torch_clone")
SHAPES=((32,128),(32,256),(32,512),(32,1024),(32,2048),(32,4096),(32,8192),(1,1024),(128,1024),(1024,1024),(4096,1024),(4096,8192))
FULL=tuple((b,d) for b in (1,32,128,1024,4096) for d in (128,256,512,1024,2048,4096,8192))

def args():
 p=argparse.ArgumentParser(); p.add_argument("--matrix",choices=("representative","full"),default="representative"); p.add_argument("--batch",type=int); p.add_argument("--dim",type=int)
 p.add_argument("--dtype",choices=(*DTYPES,"all"),default="all"); p.add_argument("--implementation",choices=(*IMPLS,"all"),default="all")
 p.add_argument("--warmup",type=int,default=100); p.add_argument("--repeat",type=int,default=1000); p.add_argument("--trials",type=int,default=3); p.add_argument("--profile",action="store_true")
 p.add_argument("--output",type=Path,default=Path("results/pytorch_comparison")); p.add_argument("--library",type=Path,default=ROOT/"build/libfused_hadamard_quant.so")
 a=p.parse_args();
 if (a.batch is None)!=(a.dim is None): p.error("--batch and --dim must be supplied together")
 if a.profile:
  if a.batch is None or a.dim is None or a.implementation=="all": p.error("profile requires one shape and implementation")
  a.repeat=1; a.trials=1
 return a

class Native:
 def __init__(self,path):
  self.lib=ctypes.CDLL(str(path)); self.lib.fused_hq_workspace_bytes.argtypes=[ctypes.c_int,ctypes.c_int64,ctypes.c_int]; self.lib.fused_hq_workspace_bytes.restype=ctypes.c_size_t
  self.lib.fused_hq_launch.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]; self.lib.fused_hq_launch.restype=ctypes.c_int
  self.lib.fused_hq_cuda_error_string.argtypes=[ctypes.c_int]; self.lib.fused_hq_cuda_error_string.restype=ctypes.c_char_p
 def prepare(self,name,x,dtype_id):
  b,d=x.shape; q=torch.empty_like(x,dtype=torch.int8); am=torch.empty(b,device=x.device); sc=torch.empty(b,device=x.device)
  n=self.lib.fused_hq_workspace_bytes(VERSIONS[name],b,d); work=torch.empty(n,device=x.device,dtype=torch.uint8)
  def launch():
   stream=torch.cuda.current_stream(x.device).cuda_stream; wp=work.data_ptr() if n else 0
   e=self.lib.fused_hq_launch(VERSIONS[name],x.data_ptr(),q.data_ptr(),am.data_ptr(),sc.data_ptr(),wp,b,d,dtype_id,stream)
   if e: raise RuntimeError(self.lib.fused_hq_cuda_error_string(e).decode())
   return q,am,sc
  return launch

def quantize(y):
 z=y.float(); am=z.abs().amax(dim=1); sc=torch.where(am>0,am/127.0,torch.ones_like(am)); q=torch.round(z/sc[:,None]).clamp(-127,127).to(torch.int8); return q,am,sc
compiled_quant=torch.compile(quantize,fullgraph=True,dynamic=True)
def reference(x):
 v=x.float().clone(); b,d=v.shape; stride=1
 while stride<d:
  z=v.view(b,d//(2*stride),2,stride); lo=z[:,:,0,:].clone(); hi=z[:,:,1,:].clone(); z[:,:,0,:]=lo+hi; z[:,:,1,:]=lo-hi; stride*=2
 return quantize(v*(1/math.sqrt(d)))
def input_for(pattern,b,d,dtype):
 if pattern=="random":
  g=torch.Generator(device="cuda"); g.manual_seed(20260920+d); return torch.randn(b,d,device="cuda",dtype=dtype,generator=g)
 if pattern=="basis":
  x=torch.zeros(b,d,device="cuda",dtype=dtype); x[torch.arange(b,device="cuda"),torch.arange(b,device="cuda")%d]=1; return x
 if pattern=="sequence": return (((torch.arange(d,device="cuda")%31)-15)/16).repeat(b,1).to(dtype)
 return torch.zeros(b,d,device="cuda",dtype=dtype)
def errs(a,b):
 q=max(abs(int(x)-int(y)) for x,y in zip(a[0].cpu().flatten().tolist(),b[0].cpu().flatten().tolist()))
 ae=(a[1]-b[1]).abs().max().item(); se=(a[2]-b[2]).abs().max().item(); ar=(a[1]-b[1]).abs().div(b[1].abs().clamp_min(1e-6)).max().item(); sr=(a[2]-b[2]).abs().div(b[2].abs().clamp_min(1e-6)).max().item()
 return q,ae,se,ar,sr
def timed(op,warmup,repeat,profile):
 for _ in range(warmup): op()
 torch.cuda.synchronize(); start=torch.cuda.Event(True); stop=torch.cuda.Event(True)
 if profile: torch.cuda.cudart().cudaProfilerStart()
 start.record();
 for _ in range(repeat): op()
 stop.record(); stop.synchronize()
 if profile: torch.cuda.cudart().cudaProfilerStop()
 return start.elapsed_time(stop)*1000/repeat

def main():
 a=args(); a.output.mkdir(parents=True,exist_ok=True); native=Native(a.library); shapes=((a.batch,a.dim),) if a.batch else (FULL if a.matrix=="full" else SHAPES); dtnames=DTYPES if a.dtype=="all" else (a.dtype,); selected=IMPLS if a.implementation=="all" else (a.implementation,)
 raw=[]; correctness=[]
 for dn in dtnames:
  dtype,did=DTYPES[dn]
  for b,d in shapes:
   x=input_for("random",b,d,dtype).contiguous(); scale=1/math.sqrt(d); native_ops={n:native.prepare(n,x,did) for n in VERSIONS}
   eager=lambda:quantize(hadamard_transform(x,scale=scale)); comp=lambda:compiled_quant(hadamard_transform(x,scale=scale)); clone=lambda:x.clone()
   ops={**native_ops,"pytorch_eager":eager,"pytorch_compiled":comp,"torch_clone":clone}; eager(); comp(); torch.cuda.synchronize()
   gates={n:True for n in IMPLS}; maxerr={n:[0,0.,0.,0.,0.] for n in IMPLS}
   for pat in ("random","basis","sequence","zero"):
    cx=input_for(pat,min(b,4),d,dtype).contiguous(); ref=reference(cx); cops={n:native.prepare(n,cx,did) for n in VERSIONS}; cs=1/math.sqrt(d)
    results={n:op() for n,op in cops.items()}; results["pytorch_eager"]=quantize(hadamard_transform(cx,scale=cs)); results["pytorch_compiled"]=compiled_quant(hadamard_transform(cx,scale=cs)); torch.cuda.synchronize()
    for n,r in results.items():
     e=errs(r,ref); maxerr[n]=[max(x,y) for x,y in zip(maxerr[n],e)]
     atol,rtol=(2e-5,1e-4) if dn=="fp32" else ((3e-3,2e-3) if dn=="fp16" else (3e-2,2e-2))
     gates[n]&=e[0]<=1 and (e[1]<=atol or e[3]<=rtol) and (e[2]<=atol/127 or e[4]<=rtol)
   cross=all(gates[n] for n in (*VERSIONS,"pytorch_eager","pytorch_compiled"));
   for n in (*VERSIONS,"pytorch_eager","pytorch_compiled"):
    correctness.append(dict(implementation=n,batch=b,head_dim=d,dtype=dn,correct=gates[n],comparison_valid=cross,max_quant_error=maxerr[n][0],max_amax_abs_error=maxerr[n][1],max_scale_abs_error=maxerr[n][2],max_amax_relative_error=maxerr[n][3],max_scale_relative_error=maxerr[n][4]))
   for trial in range(a.trials):
    order=list(IMPLS[trial%len(IMPLS):]+IMPLS[:trial%len(IMPLS)])
    for oi,n in enumerate(order):
     if n not in selected: continue
     lat=timed(ops[n],a.warmup,a.repeat,a.profile); raw.append(dict(trial=trial+1,order_index=oi,implementation=n,batch=b,head_dim=d,dtype=dn,latency_us=lat,correct=(True if n=="torch_clone" else gates[n]),comparison_valid=cross))
   print(f"[{b},{d}] {dn}: correctness={'PASS' if cross else 'FAILED'}")
 fields=list(raw[0]);
 with open(a.output/"raw.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(raw)
 with open(a.output/"correctness.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=correctness[0]);w.writeheader();w.writerows(correctness)
 meta={"created_utc":datetime.now(timezone.utc).isoformat(),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"shapes":shapes,"dtypes":list(dtnames),"warmup":a.warmup,"repeat":a.repeat,"trials":a.trials,"contract":"q=round(clamp((x@H/sqrt(D))/(rowwise_amax/127),-127,127)); zero scale=1"}; (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
 summarize(a.output,raw,correctness); return 0 if all(r["correct"] for r in correctness) else 2

def summarize(out,raw,corr):
 groups={}
 for r in raw:groups.setdefault((r["implementation"],r["batch"],r["head_dim"],r["dtype"]),[]).append(r)
 rows=[]
 for k,rs in groups.items():
  xs=[r["latency_us"] for r in rs]; med=statistics.median(xs); rows.append(dict(implementation=k[0],batch=k[1],head_dim=k[2],dtype=k[3],latency_us=med,min_us=min(xs),max_us=max(xs),variation=(max(xs)-min(xs))/med,unstable=(max(xs)-min(xs))/med>0.05,correct=all(r["correct"] for r in rs),comparison_valid=all(r["comparison_valid"] for r in rs)))
 by={(r["batch"],r["head_dim"],r["dtype"],r["implementation"]):r for r in rows}
 for r in rows:
  def ratio(name):
   z=by.get((r["batch"],r["head_dim"],r["dtype"],name)); return z["latency_us"]/r["latency_us"] if z and r["comparison_valid"] else ""
  r["speed_vs_pytorch_eager"]=ratio("pytorch_eager"); r["speed_vs_pytorch_compiled"]=ratio("pytorch_compiled"); r["speedup_vs_separate"]=ratio("separate"); r["ratio_vs_clone"]=(r["latency_us"]/by[(r["batch"],r["head_dim"],r["dtype"],"torch_clone")]["latency_us"] if (r["batch"],r["head_dim"],r["dtype"],"torch_clone") in by else "")
  for suffix in ("pytorch_eager","pytorch_compiled"):
   speed=r[f"speed_vs_{suffix}"]
   r[f"latency_ratio_vs_{suffix}"]=1/speed if isinstance(speed,float) else ""
   r[f"percent_latency_gap_vs_{suffix}"]=(1/speed-1)*100 if isinstance(speed,float) else ""
 fields=list(rows[0]);
 with open(out/"summary.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:(r["dtype"],r["batch"],r["head_dim"],r["implementation"])))
 report(out,rows)
def report(out,rows):
 lines=["# Fused Hadamard + INT8 vs PyTorch", "", "Mathematics: normalized Sylvester/Walsh Hadamard followed by rowwise dynamic symmetric INT8. CUDA-event latency uses three-run medians; allocation/input construction/correctness are outside timing. PyTorch eager includes Dao FHT and every standard quantization tensor operation; compiled compiles only quantization.",""]
 for dt in ("fp32","fp16","bf16"):
  lines += [f"## {dt.upper()}","","| Shape | Implementation | us | Speed vs eager | Speed vs compiled | Speedup vs separate | Stable | Correct |","|---|---|---:|---:|---:|---:|---|---|"]
  for r in sorted((x for x in rows if x["dtype"]==dt),key=lambda x:(x["batch"],x["head_dim"],x["implementation"])):
   fmt=lambda x: f"{x:.3f}x" if isinstance(x,float) else "—"; lines.append(f'| [{r["batch"]},{r["head_dim"]}] | {r["implementation"]} | {r["latency_us"]:.3f} | {fmt(r["speed_vs_pytorch_eager"])} | {fmt(r["speed_vs_pytorch_compiled"])} | {fmt(r["speedup_vs_separate"])} | {"UNSTABLE" if r["unstable"] else "yes"} | {"PASS" if r["correct"] else "FAIL"} |')
  lines.append("")
 lines += ["## Aggregate",""]
 for v in ("fused_v0","fused_v1","fused_v2"):
  vr=[r for r in rows if r["implementation"]==v];
  for baseline,key in (("eager","speed_vs_pytorch_eager"),("compiled","speed_vs_pytorch_compiled")):
   vals=[r[key] for r in vr if isinstance(r[key],float)]
   if vals:
    gm=math.exp(sum(math.log(x) for x in vals)/len(vals)); lines.append(f"- {v} vs PyTorch {baseline}: geometric mean {gm:.3f}x, wins {sum(x>1 for x in vals)}/{len(vals)}.")
 lines += ["","## Numerical note","","Dao returns FP16/BF16 Hadamard output in the input dtype before FP32 quantization, while fused kernels keep butterfly intermediates in FP32. Ratios are valid only when final INT8, amax, and scale pass the recorded gate.",""]
 (out/"report.md").write_text("\n".join(lines))
if __name__=="__main__": raise SystemExit(main())
