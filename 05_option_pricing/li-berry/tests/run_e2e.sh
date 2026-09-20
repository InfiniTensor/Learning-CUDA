#!/usr/bin/env bash
# run_e2e.sh - GPU 端到端回归测试（16 组用例，目标 < 30s）
# 黄金值：data/reference_values.txt（解析公式 + Python 交叉验证）
set -u
cd "$(dirname "$0")/.."
BIN=./build/option_pricer
source scripts/check.sh

EC=configs/option_european_call.txt
EP=configs/option_european_put.txt
AC=configs/option_asian_call.txt
BC=configs/option_barrier_call.txt

# ---- 1. 欧式看涨 vs Black-Scholes（|z|<3.5）----
$BIN --option $EC --sim configs/sim_default.txt --quiet
Z=$(extract results/pricing_result.txt gpu_z_score)
check "european_call vs BS (z=$Z)" "num_ok $Z 0 3.5"

# ---- 2. 欧式看跌 vs Black-Scholes ----
$BIN --option $EP --sim configs/sim_default.txt --quiet
Z=$(extract results/pricing_result.txt gpu_z_score)
check "european_put vs BS (z=$Z)" "num_ok $Z 0 3.5"

# ---- 3. 确定性：同配置重跑结果文件 byte 级一致 ----
$BIN --option $EC --sim configs/sim_default.txt --quiet
cp results/pricing_result.txt /tmp/det_a.txt
$BIN --option $EC --sim configs/sim_default.txt --quiet
check "determinism (bitwise rerun)" "diff -q /tmp/det_a.txt results/pricing_result.txt"

# ---- 4. CPU/GPU 一致性（custom RNG，同采样条件）----
$BIN --option $AC --sim configs/sim_small.txt --mode both --quiet 2>/dev/null
PG=$(extract results/pricing_result.txt gpu_price_estimate)
PC=$(extract results/pricing_result.txt cpu_price_estimate)
SG=$(extract results/pricing_result.txt gpu_mc_standard_error)
TOL=$(awk -v p="$PG" -v s="$SG" 'BEGIN { t = p * 1e-4; h = 0.5 * s; print (t > h ? t : h) }')
check "cpu vs gpu consistency (diff tol=$TOL)" "num_ok $PG $PC $TOL"

# ---- 5. 收敛：SE 斜率 ≈ -0.5（确定性），每点 |z|<3.5 ----
SWEEP_OUT=$($BIN --option $EC --sim configs/sim_default.txt --sweep 2>/dev/null)
SLOPE=$(echo "$SWEEP_OUT" | awk '/se_convergence_slope/ { print $3 }')
check "convergence: se slope = $SLOPE" "num_ok $SLOPE -0.5 0.05"
ZOK=1
while read -r line; do
    case "$line" in \#*|"") continue ;; esac
    z=$(echo "$line" | awk '{ print $5 }')
    num_ok "$z" 0 3.5 || ZOK=0
done <<< "$SWEEP_OUT"
check "convergence: all sweep |z| < 3.5" "test $ZOK -eq 1"

# ---- 6. 网格不变性：block/grid 变化价格一致 ----
PRICES=""
for cfg in "--block 128" "--block 256" "--block 512" "--block 256 --grid 1" "--block 256 --grid 576"; do
    $BIN --option $EC --sim configs/sim_default.txt $cfg --quiet
    P=$(extract results/pricing_result.txt gpu_price_estimate)
    Z=$(extract results/pricing_result.txt gpu_z_score)
    num_ok "$Z" 0 3.5 || PRICES="BROKEN"
    PRICES="$PRICES $P"
done
SPREAD=$(echo "$PRICES" | awk '{ mn = $1; mx = $1; for (i = 2; i <= NF; i++) { if ($i < mn) mn = $i; if ($i > mx) mx = $i } print (mx - mn) }')
check "grid invariance (spread=$SPREAD)" "num_ok $SPREAD 0 0.001"

# ---- 7. 亚式原始 MC：5.2995 ± 0.03 ----
$BIN --option $AC --sim configs/sim_default.txt --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
check "asian raw price = $P" "range_ok $P 5.2695 5.3295"

# ---- 8. 亚式 + 控制变量：5.2995 ± 0.005，VR > 100 ----
$BIN --option $AC --sim configs/sim_control_variate.txt --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
VR=$(extract results/pricing_result.txt gpu_variance_reduction_factor)
check "asian + cv price = $P (vr=$VR)" "range_ok $P 5.2945 5.3045"
check "asian + cv vr = $VR > 100" "num_ok $VR 1000 900"

# ---- 9. 亚式 + 对偶变量：VR > 1.1 ----
$BIN --option $AC --sim configs/sim_antithetic.txt --quiet
VR=$(extract results/pricing_result.txt gpu_variance_reduction_factor)
Z=$(extract results/pricing_result.txt gpu_z_score)
check "asian + antithetic vr = $VR" "num_ok $VR 2 0.9"
check "asian + antithetic z = $Z" "num_ok $Z 0 3.5"

# ---- 10. 障碍原始离散监控（无 BGK）：≈1.30-1.33，记录离散偏差 ----
$BIN --option $BC --sim configs/sim_default.txt --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
check "barrier raw discrete price = $P" "range_ok $P 1.25 1.37"

# ---- 11. 障碍 + BGK：1.1554 ± 0.02（残差偏差已在报告说明）----
$BIN --option $BC --sim configs/sim_default.txt --bgk --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
check "barrier + bgk price = $P" "range_ok $P 1.1354 1.1754"

# ---- 12. 障碍 + BGK + 控制变量：同价格带，VR 无损害（≥0.9）----
$BIN --option $BC --sim configs/sim_control_variate.txt --bgk --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
VR=$(extract results/pricing_result.txt gpu_variance_reduction_factor)
check "barrier + bgk + cv price = $P" "range_ok $P 1.1354 1.1754"
check "barrier + cv vr = $VR (documented ~1.0)" "num_ok $VR 1 0.1"

# ---- 13. 障碍 + BGK + 对偶 ----
$BIN --option $BC --sim configs/sim_antithetic.txt --bgk --quiet
P=$(extract results/pricing_result.txt gpu_price_estimate)
VR=$(extract results/pricing_result.txt gpu_variance_reduction_factor)
check "barrier + bgk + antithetic price = $P" "range_ok $P 1.1354 1.1754"
check "barrier + antithetic vr = $VR" "num_ok $VR 1.15 0.5"

# ---- 14. RNG 流：CPU/GPU custom uniform bitwise 一致 ----
$BIN --option $EC --sim configs/sim_consistency.txt --mode cpu --dump-rng 256 > /tmp/rng_c.txt 2>/dev/null
$BIN --option $EC --sim configs/sim_consistency.txt --mode gpu --dump-rng 256 > /tmp/rng_g.txt 2>/dev/null
check "rng streams bitwise identical" "diff -q /tmp/rng_c.txt /tmp/rng_g.txt"

# ---- 15. 步进欧式（--euro-steps 256）与精确终值一致 ----
$BIN --option $EC --sim configs/sim_default.txt --euro-steps 256 --quiet
Z=$(extract results/pricing_result.txt gpu_z_score)
P=$(extract results/pricing_result.txt gpu_price_estimate)
check "stepwise european z = $Z (price=$P)" "num_ok $Z 0 3.5"

# ---- 16. 错误处理：坏配置 → exit 1 且带行号报错 ----
cat > /tmp/bad_option.txt <<'EOF'
option_type = european_call
spot = 100
strike = 100
risk_free_rate = 0.03
volatility = 0.2
maturity = 1
barbell = 42
EOF
$BIN --option /tmp/bad_option.txt --sim configs/sim_default.txt --quiet > /dev/null 2>/tmp/err.txt
RC=$?
check "unknown key rejected (exit=$RC)" "test $RC -eq 1"
check "error message has line number" "grep -q 'line' /tmp/err.txt"

cat > /tmp/bad_barrier.txt <<'EOF'
option_type = barrier_call
spot = 100
strike = 100
risk_free_rate = 0.03
volatility = 0.2
maturity = 1
EOF
$BIN --option /tmp/bad_barrier.txt --sim configs/sim_default.txt --quiet > /dev/null 2>&1
check "barrier without barrier key rejected (exit=$?)" "test $? -eq 1"

cat > /tmp/bad_sim.txt <<'EOF'
num_paths = 99999
num_steps = 256
seed = 1
rng = "curand"
variance_reduction = "antithetic"
EOF
$BIN --option $EC --sim /tmp/bad_sim.txt --quiet > /dev/null 2>&1
check "odd paths + antithetic rejected (exit=$?)" "test $? -eq 1"

summary
