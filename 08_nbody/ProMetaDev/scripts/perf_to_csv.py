#!/usr/bin/env python3
"""
perf_to_csv.py —— 把 outputs/perf.log 的 JSON 行转成带表头的 CSV，方便 Pandas/Excel 画图。

Schema 字段 (全部可选, 缺失填空字符串):
  mode, integrator, kernel, N, steps, dt, eps, record_interval,
  total_ms, avg_step_ms, throughput_particles_per_s,
  gpu_mem_MB, cpu_time_ms, speedup, traj_file, particles_file, comment,
  raw_line

用法:
  python3 scripts/perf_to_csv.py outputs/perf.log -o outputs/perf_summary.csv
  python3 scripts/perf_to_csv.py outputs/perf.log --sort throughput --descend -n 5
  cat outputs/perf.log | python3 scripts/perf_to_csv.py - -
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
from typing import Any, Dict, List

SCHEMA_COLS = [
    "mode","integrator","kernel","N","steps","dt","eps","record_interval",
    "total_ms","avg_step_ms","throughput_particles_per_s","gpu_mem_MB",
    "cpu_time_ms","speedup","traj_file","particles_file","comment","raw_line",
]

SCHEMA_ALIASES = {
    # 键名别名 (用于从旧版 perf.log 兼容解析)
    "throughput": "throughput_particles_per_s",
    "psteps_per_s": "throughput_particles_per_s",
    "particle_steps_per_s": "throughput_particles_per_s",
    "throughput_psteps_s":"throughput_particles_per_s",
    "gpu_mem_mb":"gpu_mem_MB",
    "gpu_mem_est_mb":"gpu_mem_MB",
    "gpu_mem":"gpu_mem_MB",
    "total_time_ms":"total_ms",
    "avg_step":"avg_step_ms",
    "avg_step_time_ms":"avg_step_ms",
    "num_steps":"steps",
    "softening":"eps",
    "softening_eps":"eps",
    "kernel_mode":"kernel",
}

def normalize(obj: Dict[str,Any]) -> Dict[str,Any]:
    out: Dict[str,Any] = {c:"" for c in SCHEMA_COLS}
    # 先拷贝并应用别名
    flat: Dict[str,Any] = {}
    def walk(d: Dict[str,Any]):
        for k,v in d.items():
            if isinstance(v, dict): walk(v)
            else: flat[str(k)] = v
    walk(obj)
    for k,v in flat.items():
        key = SCHEMA_ALIASES.get(k, k)
        if key in out and out[key] == "":
            out[key] = v
    # 缺 throughput 尝试计算
    if out["throughput_particles_per_s"] == "" and out.get("N","") != "" and out.get("steps","") != "" and out.get("total_ms","") != "":
        try:
            N=float(out["N"]); steps=float(out["steps"]); t_ms=float(out["total_ms"])
            if t_ms>0: out["throughput_particles_per_s"] = (N*steps)/(t_ms*1e-3)
        except Exception: pass
    out["raw_line"] = json.dumps(obj, ensure_ascii=False)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", nargs="?", default=None, help="perf.log 路径; 用 - 表示 stdin (仅 --schema 时可省略)")
    ap.add_argument("-o","--output", default="-", help="输出 CSV 路径 (默认 - 即 stdout)")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("-n", type=int, default=-1, help="仅输出前 N 行 (header 除外)")
    ap.add_argument("--sort", default="", help=f"按某列排序, 可选列: {', '.join(c for c in SCHEMA_COLS if c!='raw_line')}")
    ap.add_argument("--descend", action="store_true")
    ap.add_argument("--schema", action="store_true", help="仅打印 schema 列定义后退出")
    args = ap.parse_args()
    if args.schema:
        print("Columns (in-order in CSV output):")
        for i,c in enumerate(SCHEMA_COLS,1):
            print(f"  {i:2d}. {c}")
        print("\nAccepted key aliases in JSON (old->new):")
        for k,v in SCHEMA_ALIASES.items(): print(f"   {k:32s} -> {v}")
        return
    if not args.input:
        ap.error("the following arguments are required: input (unless --schema is specified)")

    if args.input == "-":
        text = sys.stdin.read()
    else:
        p = pathlib.Path(args.input)
        if not p.exists():
            sys.stderr.write(f"[ERR] input not found: {args.input}\n"); sys.exit(1)
        text = p.read_text(encoding="utf-8", errors="ignore")

    rows: List[Dict[str,Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(normalize(obj))
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[warn] skip non-JSON line {line_no}: {e}\n")
            raw_dict: Dict[str,Any] = {}
            rows.append(normalize({"_parse_error": str(e), "raw_line": line}))

    if args.sort:
        if args.sort not in SCHEMA_COLS:
            sys.stderr.write(f"[ERR] --sort column '{args.sort}' not in schema (try --schema)\n"); sys.exit(2)
        def key(r):
            v = r[args.sort]
            try: return (0, float(v))
            except Exception: return (1, str(v))
        rows.sort(key=key, reverse=args.descend)

    if args.n > 0: rows = rows[:args.n]

    def wopen():
        if args.output == "-": return sys.stdout
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        return open(args.output, "w", encoding="utf-8", newline="")
    with wopen() as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA_COLS, delimiter=args.delimiter,
                           extrasaction="ignore", lineterminator="\n",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            # 把任何 list/dict 强制转字符串, 防止 csv writer 报错
            clean = {c: (r[c] if isinstance(r[c], (str,int,float,type(None))) else json.dumps(r[c],ensure_ascii=False)) for c in SCHEMA_COLS}
            w.writerow(clean)

    if args.output != "-":
        sys.stderr.write(f"[perf_to_csv] wrote {len(rows)} rows to {args.output}\n")

if __name__ == "__main__":
    main()
