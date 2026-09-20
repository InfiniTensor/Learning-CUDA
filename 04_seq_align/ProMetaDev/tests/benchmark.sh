#!/bin/bash
# 性能基准：多规模下 GPU 与 CPU（同算法）的索引构建 / 比对耗时对比
#   用法: bash tests/benchmark.sh [SM_ARCH]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SM_ARCH="${1:-90}"
BIN="$ROOT/build/seqalign"
GEN="$ROOT/tools/gen_data.py"
TMP="$ROOT/tests/tmp"
mkdir -p "$TMP"
cd "$ROOT"

echo "构建 (sm_${SM_ARCH}) ..."
bash build.sh "$SM_ARCH" >/dev/null 2>&1

printf "%-26s %8s %10s %10s %12s %12s %10s\n" \
    "规模(参考bp/reads)" "条目数" "GPU索引ms" "GPU比对ms" "CPU索引ms" "CPU比对ms" "比对加速比"
echo "-----------------------------------------------------------------------------------------------"

bench() {
    local name="$1" seqs="$2" reflen="$3" nreads="$4"
    local ref="$TMP/${name}.fa" rd="$TMP/${name}.fq"
    if [ ! -f "$ref" ] || [ ! -f "$rd" ]; then
        python3 "$GEN" --ref-seqs "$seqs" --ref-len "$reflen" --reads "$nreads" \
            --read-len 150 --sub-rate 0.03 --random-frac 0.2 --seed 2026 \
            --out-ref "$ref" --out-reads "$rd" --out-truth "$TMP/${name}.truth" >/dev/null
    fi

    local go cgo
    go=$("$BIN" gpu "$ref" "$rd" "$TMP/${name}.gpu.txt" --k 15 --band 8 --seed-step 8 2>&1)
    cgo=$("$BIN" cpuseed "$ref" "$rd" "$TMP/${name}.cpuseed.txt" --k 15 --band 8 --seed-step 8 2>&1)

    local gi=$(echo "$go" | sed -n 's/.*T_index * = \([0-9.]*\) ms/\1/p')
    local ga=$(echo "$go" | sed -n 's/.*T_align * = \([0-9.]*\) ms/\1/p')
    local ci=$(echo "$cgo" | sed -n 's/.*T_index = \([0-9.]*\) ms/\1/p')
    local ca=$(echo "$cgo" | sed -n 's/.*T_align = \([0-9.]*\) ms/\1/p')
    local ent=$(echo "$go" | sed -n 's/.*索引条目 = \([0-9]*\).*/\1/p')

    local sp=$(python3 -c "print(f'{$ca/$ga:.1f}x')")
    printf "%-26s %8s %10s %10s %12s %12s %10s\n" \
        "${name}(${reflen}/${nreads})" "$ent" "$gi" "$ga" "$ci" "$ca" "$sp"
}

bench "tiny"     2   50000    5000
bench "small"    8  200000   20000
bench "large"   20 1000000   50000
bench "target"  20 5000000  100000

echo ""
echo "硬件: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "CPU 核数: $(nproc)"
# nvcc 通常不在默认 PATH 中（与 build.sh 一样显式探测），否则这里会打印空值
NVCC=""
for d in /usr/local/cuda-12* /usr/local/cuda /usr/local/cuda-11*; do
    if [ -x "$d/bin/nvcc" ]; then NVCC="$d/bin/nvcc"; break; fi
done
if [ -n "$NVCC" ]; then
    echo "CUDA: $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
else
    echo "CUDA: N/A（未找到 nvcc）"
fi
