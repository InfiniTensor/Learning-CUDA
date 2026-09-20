#!/usr/bin/env bash
# 端到端测试：三种数据分布 × 多组配置，校验
#   1) 编解码自检
#   2) CPU 参考 vs GPU 逐位一致
#   3) 权重文件可重载反量化
#   4) 误差 / 压缩率 / 带宽指标输出
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${BIN:-./build/lowprec}"
OUT="${OUT:-outputs}"
ROWS="${ROWS:-2048}"
COLS="${COLS:-2048}"

mkdir -p "$OUT/data" "$OUT/results"
[ -x "$BIN" ] || { echo "未找到可执行文件 $BIN，请先执行 bash build.sh"; exit 2; }

echo "=== 0. 编解码自检 ==="
if ! "$BIN" selftest; then echo "自检失败"; exit 3; fi
echo

PASS=0; FAIL=0

run_case() {
    local kind="$1" cfg="$2" tag="$3"
    local t="$OUT/data/${tag}_${kind}.bin"
    local p="$OUT/results/${tag}_${kind}"
    if ! "$BIN" gen "$kind" "$ROWS" "$COLS" "$t" >/dev/null 2>&1; then
        echo "  [FAIL] gen $tag / $kind"; FAIL=$((FAIL+1)); return
    fi
    "$BIN" quant "$t" "$cfg" "$p" >/dev/null 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        PASS=$((PASS+1)); echo "  [PASS] $tag / $kind"
    else
        FAIL=$((FAIL+1)); echo "  [FAIL] $tag / $kind  (rc=$rc)"
    fi
}

CONFIGS=(
    "mxfp8_block_fp16"
    "mxfp8_block_fp16_stoch"
    "mxfp8_tensor_fp16"
    "mxfp8_block_e5m2_fp16"
    "nvfp4_block_fp16"
    "nvfp4_block_fp16_stoch"
    "nvfp4_block_fp32"
    "nvfp4_tensor_fp16"
)
KINDS=(random normal outlier)

echo "=== 1. fp32 输入：三种分布 × 多配置 ==="
for tag in "${CONFIGS[@]}"; do
    for k in "${KINDS[@]}"; do
        run_case "$k" "configs/${tag}.cfg" "$tag"
    done
done

echo
echo "=== 2. fp16 输入路径 ==="
t="$OUT/data/fp16_random.bin"
"$BIN" gen random 1024 1024 "$t" --fp16 >/dev/null 2>&1
"$BIN" quant "$t" configs/nvfp4_block_fp16.cfg "$OUT/results/fp16_nvfp4" >/dev/null 2>&1
if [ $? -eq 0 ]; then PASS=$((PASS+1)); echo "  [PASS] fp16 输入 / nvfp4"; else FAIL=$((FAIL+1)); echo "  [FAIL] fp16 输入 / nvfp4"; fi

echo
echo "=== 3. 权重文件重载反量化 (dequant 模式) ==="
q="$OUT/results/nvfp4_block_fp16_random.quant"
if [ -f "$q" ]; then
    "$BIN" dequant "$q" "$OUT/results/reload_nvfp4_dequant.bin" >/dev/null 2>&1 \
        && { PASS=$((PASS+1)); echo "  [PASS] dequant 模式"; } \
        || { FAIL=$((FAIL+1)); echo "  [FAIL] dequant 模式"; }
fi

echo
echo "=== 4. 第三方标准实现交叉验证 (ml_dtypes) ==="
if python3 -c "import ml_dtypes, numpy" 2>/dev/null; then
    cross() {
        if python3 tools/cross_check.py "$1" "$2" > /tmp/cross_one.log 2>&1; then
            PASS=$((PASS+1)); echo "  [PASS] $3"
        else
            FAIL=$((FAIL+1)); echo "  [FAIL] $3"; sed 's/^/      /' /tmp/cross_one.log
        fi
    }
    cross "$OUT/data/mxfp8_block_fp16_random.bin"  "$OUT/results/mxfp8_block_fp16_random.quant"  "mxfp8 E4M3 vs ml_dtypes"
    cross "$OUT/data/mxfp8_block_e5m2_fp16_random.bin" "$OUT/results/mxfp8_block_e5m2_fp16_random.quant" "mxfp8 E5M2 vs ml_dtypes"
    cross "$OUT/data/nvfp4_block_fp16_random.bin"  "$OUT/results/nvfp4_block_fp16_random.quant"  "nvfp4 E2M1 vs 标准幅值表"
else
    echo "  跳过（未安装 ml_dtypes：python3 -m pip install ml_dtypes）"
fi

echo
echo "=== 5. 边界与鲁棒性（非整除元素数 / 非标准 block_size / 全零） ==="
# 1001000 个元素：32 与 16 均不整除，覆盖尾块 padding 路径
"$BIN" gen normal 1000 1001 "$OUT/data/nonmul.txt" > /dev/null 2>&1
for tag in mxfp8_block_fp16 nvfp4_block_fp16 mxfp8_block64_fp16 nvfp4_block64_fp16 \
           mxfp8_block128_fp16 mxfp8_tensor_fp16 nvfp4_tensor_fp16; do
    "$BIN" quant "$OUT/data/nonmul.txt" "configs/${tag}.cfg" "$OUT/results/nonmul_${tag}" \
        > /dev/null 2>&1 \
        && { PASS=$((PASS+1)); echo "  [PASS] 1001000 元素(非整除) / $tag"; } \
        || { FAIL=$((FAIL+1)); echo "  [FAIL] 1001000 元素(非整除) / $tag"; }
done
# 单元素张量
"$BIN" gen normal 1 1 "$OUT/data/one.txt" > /dev/null 2>&1
"$BIN" quant "$OUT/data/one.txt" configs/mxfp8_block_fp16.cfg "$OUT/results/one_mxfp8" \
    > /dev/null 2>&1 \
    && { PASS=$((PASS+1)); echo "  [PASS] 单元素张量 1x1"; } \
    || { FAIL=$((FAIL+1)); echo "  [FAIL] 单元素张量 1x1"; }
# 全零张量（缩放因子为 0 的边界路径）
"$BIN" gen zeros 512 512 "$OUT/data/zeros.txt" > /dev/null 2>&1
for tag in mxfp8_block_fp16 nvfp4_block_fp16; do
    "$BIN" quant "$OUT/data/zeros.txt" "configs/${tag}.cfg" "$OUT/results/zeros_${tag}" \
        > /dev/null 2>&1 \
        && { PASS=$((PASS+1)); echo "  [PASS] 全零张量 / $tag"; } \
        || { FAIL=$((FAIL+1)); echo "  [FAIL] 全零张量 / $tag"; }
done

echo
echo "=== 汇总 ==="
printf "%-34s %-14s %-14s %-10s %-9s\n" "配置" "MAE(纯量化)" "NMAE" "压缩率" "CPU=GPU"
for log in "$OUT"/results/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    mae=$(grep -m1 -E '^[[:space:]]*MAE:' "$log" | awk '{print $2}')
    nmae=$(grep -m1 -E '^[[:space:]]*NMAE:' "$log" | awk '{print $2}')
    cr=$(grep -m1 '^压缩率:' "$log" | awk '{print $2}')
    ok=$(grep -m1 'CPU 参考 vs GPU:' "$log" | grep -o '逐位一致' || echo '-')
    printf "%-34s %-14s %-14s %-10s %-9s\n" "$name" "${mae:--}" "${nmae:--}" "${cr:--}" "$ok"
done

echo
echo "PASS=$PASS  FAIL=$FAIL"
[ $FAIL -eq 0 ] && echo "全部通过" || echo "存在失败用例"
exit $([ $FAIL -eq 0 ] && echo 0 || echo 1)
