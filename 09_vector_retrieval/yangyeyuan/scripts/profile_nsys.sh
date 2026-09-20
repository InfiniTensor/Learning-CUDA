#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=${BENCH_BIN:-"$project_dir/build/vector_search_cli"}
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
profile_dir="$project_dir/results/profile_fixture"
mkdir -p "$profile_dir"

if [[ ! -f "$profile_dir/vectors.bin" ]]; then
  "$binary" generate --vectors "$profile_dir/vectors.bin" --queries "$profile_dir/queries.bin" \
    --num-vectors 65536 --num-queries 128 --dim 128 --dtype fp32 --seed 2026
fi

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

"$nsys_bin" profile --trace=cuda,osrt --sample=none --force-overwrite=true \
  --output "$project_dir/results/exact_timeline" \
  "$binary" search --vectors "$profile_dir/vectors.bin" \
  --queries "$profile_dir/queries.bin" --config "$profile_dir/exact.toml" \
  --output "$profile_dir/result.bin"
