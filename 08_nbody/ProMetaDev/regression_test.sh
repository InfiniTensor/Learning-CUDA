#!/usr/bin/env bash
# ============================================================
# regression_test.sh —— 严格断言回归 (含正确性/文件格式/性能阈值)
#  建议每次提交/发布前执行:  bash regression_test.sh
#  每个测试产出独立 log + bin, 由 scripts/regression_assert.py 做数值/格式断言。
#  最终汇总 PASS/FAIL 计数, 失败时 exit 1。
# ============================================================
set -u
cd "$(dirname "$0")"
ROOT="$PWD"
OUT="$ROOT/outputs/regression"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

PASS=0; FAIL=0
ASSERT_SCRIPT="$ROOT/scripts/regression_assert.py"

# -------- helper: run assertion & tally (NO PIPE around tally logic!) --------
# 规则: 1) 先在当前 shell 捕获 assert 输出与 rc；
#       2) 再单独 tee 打印；
#       3) 再修改 PASS/FAIL (主 shell 生效)
assert_and_tally () {
    local msg rc
    msg=$(python3 "$ASSERT_SCRIPT" "$@" 2>&1)
    rc=$?
    echo "$msg" | tee -a "$SUMMARY"
    if [ $rc -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

# -------- helper: 任意命令 + 自动 PASS/FAIL --------
# 用法: run_and_tally LABEL  <cmd with args...>
run_and_tally () {
    local label="$1"; shift
    local msg rc
    msg=$("$@" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS $label" | tee -a "$SUMMARY"
        PASS=$((PASS+1))
    else
        echo "FAIL $label  (exit=$rc)" | tee -a "$SUMMARY"
        if [ -n "$msg" ]; then echo "  >>> $msg" | head -n 5 | tee -a "$SUMMARY"; fi
        FAIL=$((FAIL+1))
    fi
}

# ============================================================
# [0/3] Precondition: build 必须成功
# ============================================================
echo "==== [0/3] 构建 (bash build.sh) ====" | tee -a "$SUMMARY"
if ! bash build.sh > "$OUT/build.log" 2>&1 ; then
    echo "[FATAL] build 失败! 见 $OUT/build.log 末尾:" | tee -a "$SUMMARY"
    tail -40 "$OUT/build.log" | tee -a "$SUMMARY"
    exit 1
fi
echo "PASS build: ./build/nbody exists" | tee -a "$SUMMARY"; PASS=$((PASS+1))

# ============================================================
# [1/3] 2body 正确性 + 稳定性 (Leapfrog, dt=0.0005, 1000步)
# ============================================================
echo ""
echo "==== [1/3] 2body: Leapfrog × 1000步 ====" | tee -a "$SUMMARY"
LOG2="$OUT/t2.log"; BIN2="$OUT/t2.bin"
./build/nbody data/particles_2body.txt data/params_default.txt "$BIN2" "$OUT/perf_t2.log" \
    --check-energy --no-cpu --integrator leapfrog --dt 0.0005 \
    --G 1 --softening 0.005 --steps 1000 --record 100 > "$LOG2" 2>&1 \
    || echo "WARN: t2 exited $?"
assert_and_tally --label "2body.bin N=2 R>=11 bytes OK" \
       --mode bin_ok --bin "$BIN2" --expectN 2 --minR 11
assert_and_tally --label "2body step-0 |P| < 1e-10" \
       --mode momentum_step0 --input "$LOG2" --rhs 1e-10
# E_drift pattern: "E_drift |ΔE/E0|: 4.689e-03" 抓到的是比例; --as-pct 后比较百分比
# dt=0.0005 + eps=0.005 实测 <0.2%; 阈值 0.5% 留足 headroom (题面要求 <1%, 回归更严格)
assert_and_tally --label "2body Energy drift < 0.5 % (1000 步)" \
       --mode rel_number --input "$LOG2" \
       --pattern "E_drift.*?[:=]\s*([\d.eE+\-]+)" --op "<" --rhs 0.5 --as-pct

# ============================================================
# [2/3] 3body + CSV 输出 500 步
# ============================================================
echo ""
echo "==== [2/3] 3body: 500步 + CSV ====" | tee -a "$SUMMARY"
LOG3="$OUT/t3.log"; BIN3="$OUT/t3.bin"; CSV3="$OUT/t3.bin.csv"
./build/nbody data/particles_3body.txt data/params_default.txt "$BIN3" "$OUT/perf_t3.log" \
    --check-energy --no-cpu --integrator leapfrog --dt 0.001 --softening 0.001 \
    --steps 500 --record 50 --csv > "$LOG3" 2>&1
assert_and_tally --label "3body.bin N=3 R>=10 bytes OK" \
       --mode bin_ok --bin "$BIN3" --expectN 3 --minR 10
# --- CSV 存在且 >1KB (用 run_and_tally 自动 PASS/FAIL) ---
run_and_tally "3body CSV exists (>1KB)" \
    python3 - "$CSV3" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
if p.exists() and p.stat().st_size > 1024:
    print(f"OK size={p.stat().st_size} bytes")
    sys.exit(0)
else:
    print(f"BAD exists={p.exists()} size={p.stat().st_size if p.exists() else -1} bytes")
    sys.exit(2)
PY
# 3body drift 往往会大些 (小质量 3 体扰动), 设为 < 2%
assert_and_tally --label "3body Energy drift < 2 % (500步)" \
       --mode rel_number --input "$LOG3" \
       --pattern "E_drift.*?[:=]\s*([\d.eE+\-]+)" --op "<" --rhs 2.0 --as-pct

# ============================================================
# [3/3] 4096_plummer: 1000步 dt=1e-3 eps=0.02 tiling 能量漂移 <1% + 吞吐 OK
# ============================================================
echo ""
echo "==== [3/3] 4096_plummer: 1000步 tiling 漂移 + bin + 吞吐 ====" | tee -a "$SUMMARY"
LOG4="$OUT/t4k.log"; BIN4="$OUT/t4k.bin"
./build/nbody data/particles_4096_plummer.txt data/params_default.txt "$BIN4" "$OUT/perf_t4k.log" \
    --check-energy --no-cpu --integrator leapfrog --kernel tiling \
    --dt 0.001 --softening 0.02 --steps 1000 --record 100 > "$LOG4" 2>&1
assert_and_tally --label "4096_plummer.bin N=4096 R=11 bytes OK" \
       --mode bin_ok --bin "$BIN4" --expectN 4096 --minR 11
assert_and_tally --label "4096_plummer Energy drift < 1 % (1000步, dt=1e-3, eps=0.02)" \
       --mode rel_number --input "$LOG4" \
       --pattern "E_drift.*?[:=]\s*([\d.eE+\-]+)" --op "<" --rhs 1.0 --as-pct
assert_and_tally --label "4096_plummer throughput > 8 M particle-steps/s" \
       --mode rel_number --input "$LOG4" \
       --pattern "throughput\s*:\s*([\d.eE+\-]+)" --op ">" --rhs 8e6

# ============================================================
# 总结
# ============================================================
echo ""
echo "==== REGRESSION SUMMARY ====" | tee -a "$SUMMARY"
echo "  PASS = $PASS" | tee -a "$SUMMARY"
echo "  FAIL = $FAIL" | tee -a "$SUMMARY"
echo "  detail log: $OUT/*.log  ;  summary: $SUMMARY" | tee -a "$SUMMARY"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "[REGRESSION FAILED] 有 $FAIL 个断言未通过! 详细见上面的 FAIL 行"
    echo "  hint: 查看 $OUT/t2.log / t3.log / t4k.log 末尾输出"
    exit 1
else
    echo "ALL REGRESSION ASSERTIONS PASSED ($PASS)." | tee -a "$SUMMARY"
    exit 0
fi
