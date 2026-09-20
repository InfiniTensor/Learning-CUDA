#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=${BENCH_BIN:-"$project_dir/build/vector_search_cli"}
result_dir=${RESULT_DIR:-"$project_dir/results"}
num_vectors=${NUM_VECTORS:-1000000}
num_queries=${NUM_QUERIES:-1000}
dim=${DIM:-128}
top_k=${TOP_K:-10}
nlist=${NLIST:-4096}
mkdir -p "$result_dir/configs"

vectors="$result_dir/vectors_n${num_vectors}_d${dim}.bin"
queries="$result_dir/queries_q${num_queries}_d${dim}.bin"
index="$result_dir/index_n${num_vectors}_d${dim}_l${nlist}.ivf"
ground_truth="$result_dir/exact_k${top_k}.bin"

if [[ ! -f "$vectors" || ! -f "$queries" ]]; then
  "$binary" generate --vectors "$vectors" --queries "$queries" \
    --num-vectors "$num_vectors" --num-queries "$num_queries" \
    --dim "$dim" --dtype fp32 --seed 2026
fi

exact_config="$result_dir/configs/exact.toml"
cat >"$exact_config" <<EOF
top_k = $top_k
search_mode = "exact"
batch_size = 128
nlist = $nlist
nprobe = 16
kmeans_iters = 12
train_size = 65536
seed = 2026
EOF
"$binary" search --vectors "$vectors" --queries "$queries" \
  --config "$exact_config" --output "$ground_truth" \
  --performance "$result_dir/performance_exact.json"

build_config="$result_dir/configs/build.toml"
cat >"$build_config" <<EOF
top_k = $top_k
search_mode = "ivf_flat"
batch_size = 128
nlist = $nlist
nprobe = 16
kmeans_iters = 12
train_size = 65536
seed = 2026
EOF
if [[ ! -f "$index" ]]; then
  "$binary" build --vectors "$vectors" --index "$index" \
    --config "$build_config" --performance "$result_dir/performance_build.json"
fi

for batch_size in 1 16 64 128; do
  for nprobe in 1 4 8 16 32 64; do
    config="$result_dir/configs/ivf_b${batch_size}_p${nprobe}.toml"
    cat >"$config" <<EOF
top_k = $top_k
search_mode = "ivf_flat"
batch_size = $batch_size
nlist = $nlist
nprobe = $nprobe
kmeans_iters = 12
train_size = 65536
seed = 2026
EOF
    stem="b${batch_size}_p${nprobe}_k${top_k}"
    "$binary" search --index "$index" --queries "$queries" \
      --config "$config" --output "$result_dir/ivf_${stem}.bin" \
      --performance "$result_dir/performance_${stem}.json" \
      --ground-truth "$ground_truth" --quality "$result_dir/quality_${stem}.json"
  done
done

echo "benchmark logs written to $result_dir"
