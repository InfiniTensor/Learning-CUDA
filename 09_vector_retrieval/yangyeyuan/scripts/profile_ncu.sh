#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=${BENCH_BIN:-"$project_dir/build/vector_search_cli"}
ncu_bin=${NCU_BIN:-/usr/local/cuda/bin/ncu}
profile_dir="$project_dir/results/profile_fixture"
mkdir -p "$profile_dir"

"$binary" generate --vectors "$profile_dir/vectors.bin" --queries "$profile_dir/queries.bin" \
  --num-vectors 65536 --num-queries 16 --dim 128 --dtype fp32 --seed 2026

cat >"$profile_dir/exact.toml" <<EOF
top_k = 10
search_mode = "exact"
batch_size = 16
nlist = 256
nprobe = 16
kmeans_iters = 4
train_size = 4096
seed = 2026
EOF

"$ncu_bin" --set full --kernel-name regex:distance_partial_topk_kernel \
  --launch-count 1 --force-overwrite \
  --export "$project_dir/results/exact_sm120" \
  "$binary" search --vectors "$profile_dir/vectors.bin" \
  --queries "$profile_dir/queries.bin" --config "$profile_dir/exact.toml" \
  --output "$profile_dir/result.bin"
