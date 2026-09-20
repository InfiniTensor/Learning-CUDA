#!/bin/bash
# 性能基准测试：多规模图性能评测
# 用法: bash benchmark.sh [SM_ARCH]

SM_ARCH="${SM_ARCH:-90}"
MAXFLOW="$(dirname "$0")/../build/maxflow"
GEN="$(dirname "$0")/../tools/gen_graph.py"
TMP="$(dirname "$0")/tmp"
mkdir -p "$TMP"

echo "构建项目 (sm_${SM_ARCH})..."
bash "$(dirname "$0")/../build.sh" "$SM_ARCH" >/dev/null 2>&1
echo ""

# 预热：build.sh 会重建二进制，首次启动内核时驱动要做一次性 PTX JIT（本机 sm_90 PTX → sm_120）。
# 不预热的话这笔开销会被算进第一行（tiny）的 TTFQ/T_total，实测能到 100 ms 量级，污染整张表。
python3 "$GEN" 20 100 "$TMP/warmup.csr" "$TMP/warmup.q" --seed 1 --queries 1 --type random >/dev/null 2>&1
"$MAXFLOW" run "$TMP/warmup.csr" "$TMP/warmup.q" "$TMP/warmup.result" >/dev/null 2>&1
rm -f "$TMP/warmup.csr" "$TMP/warmup.q" "$TMP/warmup.result"

printf "%-22s %8s %8s %12s %12s %12s %10s\n" \
    "Scale(N,M)" "Queries" "Phases" "T_preprocess" "TTFQ" "T_total" "TPQ(ms)"
printf "%s\n" "-------------------------------------------------------------------------------------------------------"

run_bench() {
    local name="$1" N="$2" M="$3" Q="$4"
    local g="$TMP/${name}.csr" q="$TMP/${name}.q" r="$TMP/${name}.result"
    python3 "$GEN" "$N" "$M" "$g" "$q" --seed 42 --queries "$Q" --type random >/dev/null 2>&1
    local out
    out=$("$MAXFLOW" run "$g" "$q" "$r" 2>&1)
    local phases=$(echo "$out" | grep "阶段数" | awk '{print $NF}')
    local tpre=$(echo "$out" | grep "T_preprocess" | awk '{print $4}')
    local ttfq=$(echo "$out" | grep "TTFQ" | awk '{print $4}')
    local ttotal=$(echo "$out" | grep "T_total" | awk '{print $4}')
    local tpq=$(echo "$out" | grep "TPQ" | awk '{print $4}')
    printf "%-22s %8s %8s %12s %12s %12s %10s\n" \
        "$name($N,$M)" "$Q" "$phases" "$tpre" "$ttfq" "$ttotal" "$tpq"
}

# 每个规模统一用 12 个查询：题目要求「支持至少 10 个不同的源汇对查询」，
# 且 T_total / TPQ 需在真实多查询场景下统计。
run_bench "tiny"      20        100    12
run_bench "small"    100        800    12
run_bench "medium"   500       4000    12
run_bench "large"   2000      20000    12
run_bench "huge"   10000     100000    12
run_bench "massive" 100000  1000000    12

echo ""
echo "硬件: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
# nvcc 通常不在默认 PATH 中（与 build.sh 一样显式探测），否则这里会打印空值
NVCC=""
for d in /usr/local/cuda-12* /usr/local/cuda /usr/local/cuda-11*; do
    if [ -x "$d/bin/nvcc" ]; then NVCC="$d/bin/nvcc"; break; fi
done
if [ -n "$NVCC" ]; then
    echo "CUDA: $("$NVCC" --version | grep -o 'release [0-9.]*' | head -1)"
else
    echo "CUDA: N/A（未找到 nvcc）"
fi
