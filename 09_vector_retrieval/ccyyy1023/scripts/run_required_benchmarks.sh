#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p data results

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

if [[ ! -f data/required_db.bin || ! -f data/required_queries.bin ]]; then
  ./build/vector_search generate \
    --db data/required_db.bin --queries data/required_queries.bin \
    --n 1000000 --nq 1000 --dim 128 --metric l2 --seed 42
fi

if [[ ! -f data/required_ivf.bin ]]; then
  ./build/vector_search build \
    --db data/required_db.bin --index data/required_ivf.bin \
    --nlist 256 --iterations 8 --seed 42 | tee results/index_build.txt
fi

for k in 1 10 50 100; do
  ./build/vector_search benchmark \
    --db data/required_db.bin --queries data/required_queries.bin \
    --index data/required_ivf.bin --log "results/k_${k}.json" \
    --k "$k" --batch 128 --nlist 256 --nprobe 16 \
    --cpu-query-limit 20 --latency-query-limit 100
done

for batch in 1 32; do
  ./build/vector_search benchmark \
    --db data/required_db.bin --queries data/required_queries.bin \
    --index data/required_ivf.bin --log "results/batch_${batch}.json" \
    --k 10 --batch "$batch" --nlist 256 --nprobe 16 \
    --cpu-query-limit 20 --latency-query-limit 100
done

for nprobe in 1 4 64; do
  ./build/vector_search benchmark \
    --db data/required_db.bin --queries data/required_queries.bin \
    --index data/required_ivf.bin --log "results/nprobe_${nprobe}.json" \
    --k 10 --batch 128 --nlist 256 --nprobe "$nprobe" \
    --cpu-query-limit 20 --latency-query-limit 100
done

python3 scripts/summarize_results.py results results/SUMMARY.md
echo "Required benchmark complete: results/SUMMARY.md"

