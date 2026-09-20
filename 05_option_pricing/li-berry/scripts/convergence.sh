#!/usr/bin/env bash
# convergence.sh - 收敛性扫描 → data/convergence.csv
# 覆盖：欧式（vs BS）、亚式原始/CV（vs 高精度参考）、障碍 BGK（vs 连续闭式解）
set -u
cd "$(dirname "$0")/.."
BIN=./build/option_pricer
mkdir -p data

OUT=data/convergence.csv
: > "$OUT"

run_sweep() { # $1=label $2=option文件 $3=sim文件 $4=额外参数...
    echo "# --- $1 ---" >> "$OUT"
    $BIN --option "$2" --sim "$3" ${4:-} --sweep 2>/dev/null >> "$OUT"
}

echo "# 收敛性扫描（N = 1e4..1e7，同 seed 嵌套前缀）" > "$OUT"

run_sweep "european_call_vs_BS" configs/option_european_call.txt configs/sim_default.txt
run_sweep "european_put_vs_BS" configs/option_european_put.txt configs/sim_default.txt
run_sweep "asian_raw" configs/option_asian_call.txt configs/sim_default.txt
run_sweep "asian_cv" configs/option_asian_call.txt configs/sim_control_variate.txt
run_sweep "barrier_bgk" configs/option_barrier_call.txt configs/sim_default.txt --bgk
run_sweep "barrier_raw" configs/option_barrier_call.txt configs/sim_default.txt

echo "输出: $OUT"
