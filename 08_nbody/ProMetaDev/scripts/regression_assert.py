#!/usr/bin/env python3
"""
regression_assert.py —— regression_test.sh 的严格断言 helper。

用法:
  python3 regression_assert.py \
      --label  "2body drift <0.1%" \
      --mode   rel_number \
      --input  "path/to/stdout.log"  \
      --pattern "E_drift.*?([\d.eE+\-]+)" \
      --op     "<" --rhs 0.001 --as-pct

  python3 regression_assert.py \
      --label "4096 bin size OK" \
      --mode  bin_ok \
      --bin outputs/t4096.bin \
      --expectN 4096 --minR 5

支持的 mode:
  - rel_number: 在 --input 文件用 --pattern 抓最后一个匹配的浮点数字，和 --rhs 做 --op {<,<=,>,>=,==,!=} 比较
      可选 --as-pct (pattern 抓到的是 |ΔE/E0| ratio, 乘以 100 后再比较, 即 --rhs 写百分比)
  - line_present: --input 文件中包含至少一行匹配 --pattern
  - bin_ok: python3 scripts/check_bin.py --bin 指定文件 OK=True, 且 N=expectN, R>=minR
  - momentum_step0: --input 里 "[E] step 0:  E=xxx  |P|=([\d.eE+\-]+)" 抓到的动量 < --rhs (默认 1e-6)

退出码: 0 = PASS, 非0 = FAIL (stderr 打印原因; stdout 打印 PASS/FAIL + label)
"""
from __future__ import annotations
import argparse, pathlib, re, subprocess, sys, os, json

def die(msg: str, label: str):
    sys.stderr.write(f"[FAIL] {label}: {msg}\n")
    sys.stdout.write(f"FAIL {label}\n")
    sys.exit(2)

def ok(label: str, detail: str = ""):
    sys.stdout.write(f"PASS {label}" + (f"  ({detail})" if detail else "") + "\n")
    sys.exit(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="assertion 名称 (用于输出)")
    ap.add_argument("--mode", required=True, choices=["rel_number","line_present","bin_ok","momentum_step0"])
    ap.add_argument("--input", help="日志文件 (mode=rel_number/line_present/momentum_step0)")
    ap.add_argument("--bin", help="二进制轨迹文件 (mode=bin_ok)")
    ap.add_argument("--pattern", help="正则 (支持 group 1 为数字)")
    ap.add_argument("--op", choices=["<","<=",">",">=","==","!="])
    ap.add_argument("--rhs", type=float)
    ap.add_argument("--as-pct", action="store_true", help="抓到的 ratio*100 后再比较 (百分比模式)")
    ap.add_argument("--expectN", type=int, help="bin_ok: 期望 N")
    ap.add_argument("--minR", type=int, default=1, help="bin_ok: R 至少 >= minR (默认 1)")
    args = ap.parse_args()

    if args.mode == "bin_ok":
        # 找 check_bin.py (与本脚本同目录)
        me = pathlib.Path(__file__).resolve().parent
        check = me / "check_bin.py"
        if not check.exists():
            die(f"cannot find scripts/check_bin.py (looked at: {check})", args.label)
        if not pathlib.Path(args.bin).exists():
            die(f"bin file not found: {args.bin}", args.label)
        r = subprocess.run(["python3", str(check), args.bin], capture_output=True, text=True, timeout=60)
        # 期望输出: "path: N=X R=Y bytes=Z expected=Z OK=True"
        out = r.stdout + r.stderr
        m = re.search(r"N=(\d+)\s+R=(\d+)\s+bytes=(\d+)\s+expected=(\d+)\s+OK=(\w+)", out)
        if not m:
            die(f"unexpected check_bin output: {out.strip()[:300]}", args.label)
        N,R,bytes_,expected,okflag = int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),m.group(5)
        if okflag != "True":
            die(f"bin check OK=False (N={N} R={R} bytes={bytes_} expected={expected})", args.label)
        if args.expectN is not None and N != args.expectN:
            die(f"bin N expect {args.expectN} got {N}", args.label)
        if R < args.minR:
            die(f"bin R expect >= {args.minR} got {R}", args.label)
        if bytes_ != expected:
            die(f"bin bytes mismatch {bytes_} != expected {expected}", args.label)
        ok(args.label, f"N={N} R={R} bytes={bytes_}")
        return

    if not args.input or not pathlib.Path(args.input).exists():
        die(f"input log file not found: {args.input}", args.label)
    text = pathlib.Path(args.input).read_text(encoding="utf-8", errors="ignore")

    if args.mode == "line_present":
        if not re.search(args.pattern, text, flags=re.S):
            die(f"pattern not found in {args.input}: {args.pattern}", args.label)
        ok(args.label)
        return

    if args.mode == "momentum_step0":
        pat = args.pattern or r"\[E\]\s+step\s+0\s*:.*?\|P\|\s*=\s*([\d.eE+\-]+)"
        m = re.search(pat, text)
        if not m:
            die(f"step-0 momentum line not found in {args.input}", args.label)
        val = float(m.group(1))
        rhs = args.rhs if args.rhs is not None else 1e-6
        if val < rhs:
            ok(args.label, f"|P|={val:.2e} < {rhs:.0e}")
        else:
            die(f"|P|={val:.2e} >= {rhs:.0e}", args.label)
        return

    # rel_number
    mlist = re.findall(args.pattern, text, flags=re.S)
    if not mlist:
        die(f"pattern matches zero in {args.input}: {args.pattern}", args.label)
    last = float(mlist[-1])
    val = last * 100.0 if args.as_pct else last
    rhs = args.rhs
    def cmp(a,b,op):
        if op=="<":  return a < b
        if op=="<=": return a <= b
        if op==">":  return a > b
        if op==">=": return a >= b
        if op=="==": return abs(a-b) < 1e-12
        if op=="!=": return abs(a-b) >= 1e-12
        raise ValueError(op)
    unit = "%" if args.as_pct else ""
    ok_ = cmp(val, rhs, args.op)
    if ok_:
        ok(args.label, f"got {val:.4g}{unit} {args.op} {rhs}{unit} (raw={last:.3g})")
    else:
        die(f"got {val:.4g}{unit}  DOES NOT HOLD  {args.op} {rhs}{unit} (raw={last:.3g})", args.label)

if __name__ == "__main__":
    main()
