#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

mkdir -p data results
./build/vector_search generate \
  --db data/small_db.bin --queries data/small_queries.bin \
  --n 10000 --nq 100 --dim 128 --metric l2 --seed 42
./build/vector_search build \
  --db data/small_db.bin --index data/small_ivf.bin \
  --nlist 64 --iterations 8 --seed 42
./build/vector_search benchmark \
  --db data/small_db.bin --queries data/small_queries.bin \
  --index data/small_ivf.bin --log results/smoke.json \
  --k 10 --batch 32 --nlist 64 --nprobe 8 \
  --cpu-query-limit 20 --latency-query-limit 50

python3 scripts/summarize_results.py results results/SMOKE_SUMMARY.md
echo "Smoke test complete: results/SMOKE_SUMMARY.md"

