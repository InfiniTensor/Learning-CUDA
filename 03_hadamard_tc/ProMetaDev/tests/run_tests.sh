#!/usr/bin/env bash
# 端到端测试：FHT 正确性（n、scale、dtype 覆盖）、融合一致性、
# 以及「Hadamard 旋转降低量化误差」的效果验证
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${BIN:-./build/hadamard}"
OUT="${OUT:-outputs/tests}"
ROWS="${ROWS:-4096}"
mkdir -p "$OUT/data" "$OUT/results"
[ -x "$BIN" ] || { echo "未找到 $BIN，请先执行 bash build.sh"; exit 2; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  [PASS] $1"; }
bad() { FAIL=$((FAIL+1)); echo "  [FAIL] $1"; }

cfg_write() {   # cfg_write <path> <n> <scale> <quant:true|false> <fmt> <block>
    cat > "$1" <<EOF
hadamard_size = $2
scale = $3
quantize_enable = $4
quant_format = "$5"
block_size = $6
EOF
}

echo "=== 0. 引擎自检 ==="
if "$BIN" selftest > /tmp/hd_st.log 2>&1; then ok "selftest"; else bad "selftest"; tail -8 /tmp/hd_st.log | sed 's/^/      /'; fi

echo
echo "=== 1. GPU FHT vs PyTorch/定义式参考：n ∈ {64,128,256} ==="
for N in 64 128 256; do
    D="$OUT/data/d$N.txt"
    "$BIN" gen normal "$ROWS" "$N" "$D" --seed 7 > /dev/null
    python3 tools/torch_ref.py "$D" "$OUT/results/ref$N.txt" --n "$N" --scale 1.0 --no-gpu > /dev/null 2>&1
    cfg_write /tmp/hd_n$N.cfg "$N" 1.0 false fp8_e4m3 32
    if "$BIN" run "$D" /tmp/hd_n$N.cfg "$OUT/results/n$N" --ref "$OUT/results/ref$N.txt" > /tmp/hd_n$N.log 2>&1; then
        ok "n=$N：与参考一致"
    else
        bad "n=$N"; grep -E 'max_abs_error|超阈值' /tmp/hd_n$N.log | head -3 | sed 's/^/      /'
    fi
done

echo
echo "=== 2. scale = 1/sqrt(n)（正交归一化）==="
for N in 64 128 256; do
    D="$OUT/data/d$N.txt"
    S=$(python3 -c "print(1.0/($N)**0.5)")
    python3 tools/torch_ref.py "$D" "$OUT/results/refs$N.txt" --n "$N" --scale "$S" --no-gpu > /dev/null 2>&1
    cfg_write /tmp/hd_s$N.cfg "$N" "$S" false fp8_e4m3 32
    if "$BIN" run "$D" /tmp/hd_s$N.cfg "$OUT/results/s$N" --ref "$OUT/results/refs$N.txt" > /tmp/hd_s$N.log 2>&1; then
        ok "n=$N scale=1/sqrt(n)：与参考一致"
    else
        bad "n=$N scale=1/sqrt(n)"
    fi
done

echo
echo "=== 3. 输入精度 fp32 / fp16 / bf16 ==="
for DT in fp32 fp16 bf16; do
    D="$OUT/data/dt_$DT.txt"
    "$BIN" gen normal "$ROWS" 128 "$D" --dtype "$DT" --seed 3 > /dev/null
    python3 tools/torch_ref.py "$D" "$OUT/results/ref_$DT.txt" --n 128 --scale 1.0 --no-gpu > /dev/null 2>&1
    if "$BIN" run "$D" /tmp/hd_n128.cfg "$OUT/results/dt_$DT" --ref "$OUT/results/ref_$DT.txt" > /tmp/hd_dt.log 2>&1; then
        ok "输入 $DT"
    else
        bad "输入 $DT"; grep -E 'max_abs_error|不足' /tmp/hd_dt.log | head -2 | sed 's/^/      /'
    fi
done

echo
echo "=== 4. 融合 kernel 与分离式结果一致性 + e4m3/e5m2 ==="
for FMT in fp8_e4m3 fp8_e5m2; do
    cfg_write /tmp/hd_f.cfg 128 1.0 true "$FMT" 32
    "$BIN" run "$OUT/data/d128.txt" /tmp/hd_f.cfg "$OUT/results/f_$FMT" > /tmp/hd_f.log 2>&1
    line=$(grep '融合 vs 分离' /tmp/hd_f.log | head -1)
    if echo "$line" | grep -q '不一致 0 / .*缩放因子不一致 0 /'; then
        ok "$FMT：融合与分离逐字节一致"
    else
        bad "$FMT：融合一致性"; echo "      $line"
    fi
done

echo
echo "=== 5. Hadamard 旋转对量化误差的影响（outlier 数据，E4M3）==="
OD="$OUT/data/outlier.txt"
"$BIN" gen outlier "$ROWS" 128 "$OD" --seed 5 > /dev/null
cfg_write /tmp/hd_q.cfg 128 1.0 true fp8_e4m3 32
"$BIN" run "$OD" /tmp/hd_q.cfg "$OUT/results/rot" > /tmp/hd_rot.log 2>&1
"$BIN" quant "$OD" /tmp/hd_q.cfg "$OUT/results/raw" > /tmp/hd_raw.log 2>&1
r_rot=$(grep -m1 'rel_L2' /tmp/hd_rot.log | grep -oP 'rel_L2\s*=\s*\K[0-9.eE+-]+')
r_raw=$(grep -m1 'rel_L2' /tmp/hd_raw.log | grep -oP 'rel_L2=\K[0-9.eE+-]+')
echo "      未旋转 rel_L2 = ${r_raw:-NA}"
echo "      旋转后 rel_L2 = ${r_rot:-NA}"
if [ -n "$r_rot" ] && [ -n "$r_raw" ] && python3 -c "import sys; sys.exit(0 if float('$r_rot') < float('$r_raw') else 1)"; then
    ok "Hadamard 旋转降低了量化误差"
else
    bad "旋转未降低量化误差（请检查数据分布）"
fi

echo
echo "=== 汇总 ==="
echo "PASS=$PASS  FAIL=$FAIL"
[ $FAIL -eq 0 ] && echo "全部通过" || echo "存在失败用例"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
