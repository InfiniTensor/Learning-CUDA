#!/usr/bin/env bash
# bench.sh - 性能基准扫描 → data/bench.csv
# 维度：block × grid × rng × 方差缩减；并对比 CPU 单线程基线（亚式 200k 路径）
set -u
cd "$(dirname "$0")/.."
BIN=./build/option_pricer
mkdir -p data

OUT=data/bench.csv
: > "$OUT"
echo "option,block,grid,rng,variance_reduction,bgk,engine,paths,time_ms,paths_per_sec,kernel_ms" > "$OUT"

bench_run() { # $1=option $2=block $3=grid $4=rng $5=vr $6=bgk $7=paths
    local SIM=data/_bench_sim.txt
    cat > "$SIM" <<EOF
num_paths = $7
num_steps = 256
seed = 1234
rng = "$4"
variance_reduction = "$5"
EOF
    $BIN --option "$1" --sim "$SIM" --block "$2" --grid "$3" $6 --quiet 2>/dev/null
    local T P K
    T=$(extract results/perf_log.txt gpu_total_runtime_ms)
    P=$(extract results/perf_log.txt gpu_paths_per_second)
    K=$(extract results/perf_log.txt gpu_kernel_time_ms)
    echo "$(basename $1 .txt),$2,$3,$4,$5,${6:-off},gpu,$7,$T,$P,$K" >> "$OUT"
}
source scripts/check.sh

echo "# GPU 粒度扫描（亚式 256 步，1M 路径）"
for b in 128 256 512 1024; do
    for g in 0 144 576 2304; do
        bench_run configs/option_asian_call.txt "$b" "$g" curand none "" 1000000
    done
done

echo "# RNG 与方差缩减对比（默认粒度，10M 路径）"
bench_run configs/option_asian_call.txt 256 0 curand none "" 10000000
bench_run configs/option_asian_call.txt 256 0 custom none "" 10000000
bench_run configs/option_asian_call.txt 256 0 curand antithetic "" 10000000
bench_run configs/option_asian_call.txt 256 0 curand control_variate "" 10000000
bench_run configs/option_barrier_call.txt 256 0 curand none "" 10000000
bench_run configs/option_barrier_call.txt 256 0 curand none --bgk 10000000
bench_run configs/option_european_call.txt 256 0 curand none "" 10000000

echo "# CPU 单线程基线（亚式/障碍 200k 路径；欧式 1M）"
for opt in option_asian_call option_barrier_call; do
    cat > data/_bench_sim.txt <<'EOF'
num_paths = 200000
num_steps = 256
seed = 1234
rng = "custom"
variance_reduction = "none"
EOF
    $BIN --option configs/$opt.txt --sim data/_bench_sim.txt --mode cpu --quiet 2>/dev/null
    T=$(extract results/perf_log.txt cpu_time_ms)
    P=$(extract results/perf_log.txt cpu_paths_per_second)
    echo "$opt,1,1,custom,none,off,cpu,200000,$T,$P,0" >> "$OUT"
done
cat > data/_bench_sim.txt <<'EOF'
num_paths = 1000000
num_steps = 256
seed = 1234
rng = "custom"
variance_reduction = "none"
EOF
$BIN --option configs/option_european_call.txt --sim data/_bench_sim.txt --mode cpu --quiet 2>/dev/null
T=$(extract results/perf_log.txt cpu_time_ms)
P=$(extract results/perf_log.txt cpu_paths_per_second)
echo "option_european_call,1,1,custom,none,off,cpu,1000000,$T,$P,0" >> "$OUT"
rm -f data/_bench_sim.txt

echo "输出: $OUT"
