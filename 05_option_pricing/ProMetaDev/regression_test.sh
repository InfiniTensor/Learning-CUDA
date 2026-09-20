#!/usr/bin/env bash
# regression_test.sh - 全面回归测试矩阵
# 测试 6 种期权类型, 验证价格/SE/Greeks/性能
set -e
cd "$(dirname "$0")"
BIN=./build/cuda_pricing
OUT=outputs/regression
mkdir -p "$OUT"

PATHS=${1:-2000000}   # 默认 2M 路径 (兼顾速度与精度)
SEED=42
STEPS=256

# 期权类型列表
TYPES="EUROPEAN_CALL EUROPEAN_PUT ASIAN_CALL ASIAN_PUT BARRIER_CALL BARRIER_PUT"

echo "============================================================"
echo "  CUDA Pricing - Regression Test Matrix"
echo "  paths=$PATHS  steps=$STEPS  seed=$SEED"
echo "============================================================"
$BIN --info
echo ""

# 生成各类型参数文件并运行
for T in $TYPES; do
    # 调整 barrier 参数: PUT 用 DOWN-and-IN
    BARRIER_DIR="UP"
    if [ "$T" = "BARRIER_PUT" ]; then BARRIER_DIR="DOWN"; fi
    BARRIER="130.0"
    if [ "$BARRIER_DIR" = "DOWN" ]; then BARRIER="80.0"; fi

    cat > /tmp/params_$T.txt << EOF
type        = $T
spot        = 100.0
strike      = 100.0
risk_free   = 0.03
volatility  = 0.2
maturity    = 1.0
barrier     = $BARRIER
barrier_dir = $BARRIER_DIR
barrier_type= KNOCK_IN
EOF

    echo "---- $T (barrier=$BARRIER dir=$BARRIER_DIR) ----"
    $BIN /tmp/params_$T.txt params/sim_params.txt \
        "$OUT/${T}.json" "$OUT/regression.log" \
        --paths $PATHS --seed $SEED --no-cpu --greeks --json 2>&1
    echo ""
done

echo ""
echo "============================================================"
echo "  Summary Table"
echo "============================================================"
printf "%-16s %10s %10s %10s %8s %8s %8s %8s %8s %8s\n" \
    "Type" "Price" "Ref" "AbsErr" "SE" "Delta" "Gamma" "Vega" "Theta" "Rho"

for T in $TYPES; do
    # 从 JSON 提取关键字段
    PRICE=$(grep '"price"' "$OUT/${T}.json" | head -1 | sed 's/.*: //' | sed 's/,//')
    REF=$(grep '"ref_price"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    SE=$(grep '"std_error"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    D=$(grep '"delta"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    G=$(grep '"gamma"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    V=$(grep '"vega"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    TH=$(grep '"theta"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/,//')
    RH=$(grep '"rho"' "$OUT/${T}.json" | sed 's/.*: //' | sed 's/[ ,]*$//')
    ERR=$(python3 -c "print(abs($PRICE - $REF))" 2>/dev/null || echo "N/A")
    printf "%-16s %10.4f %10.4f %10.4f %8.5f %8.4f %8.5f %8.3f %8.3f %8.3f\n" \
        "$T" "$PRICE" "$REF" "$ERR" "$SE" "$D" "$G" "$V" "$TH" "$RH"
done

echo ""
echo "==== CPU vs GPU Speedup (1M paths, ASIAN_CALL) ===="
$BIN params/option_params.txt params/sim_params.txt \
    "$OUT/speedup.json" "$OUT/regression.log" \
    --paths 1000000 --seed $SEED --json 2>&1 | grep -E '^\[(CPU|GPU)\]'
