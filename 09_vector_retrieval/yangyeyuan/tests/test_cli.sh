#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=${1:-"$project_dir/build/vector_search_cli"}
fixture=$(mktemp -d /tmp/vector-search-cli.XXXXXX)
trap 'rm -rf "$fixture"' EXIT

"$binary" self-test
"$binary" generate \
  --vectors "$fixture/vectors.bin" --queries "$fixture/queries.bin" \
  --num-vectors 513 --num-queries 17 --dim 32 --dtype fp16 --seed 9
"$binary" cpu \
  --vectors "$fixture/vectors.bin" --queries "$fixture/queries.bin" \
  --config "$project_dir/configs/test_ivf.toml" \
  --output "$fixture/ground_truth.bin"
"$binary" build \
  --vectors "$fixture/vectors.bin" --index "$fixture/index.ivf" \
  --config "$project_dir/configs/test_ivf.toml" \
  --performance "$fixture/build.json"
"$binary" search \
  --index "$fixture/index.ivf" --queries "$fixture/queries.bin" \
  --config "$project_dir/configs/test_ivf.toml" \
  --output "$fixture/ivf.bin" --performance "$fixture/search.json" \
  --ground-truth "$fixture/ground_truth.bin" --quality "$fixture/quality.json"

test -s "$fixture/build.json"
test -s "$fixture/search.json"
test -s "$fixture/quality.json"

cat >"$fixture/negative.toml" <<EOF
top_k = -1
search_mode = "exact"
batch_size = 7
nlist = 8
nprobe = 8
kmeans_iters = 4
train_size = 513
seed = 2026
EOF
if "$binary" cpu --vectors "$fixture/vectors.bin" --queries "$fixture/queries.bin" \
    --config "$fixture/negative.toml" --output "$fixture/invalid.bin"; then
  echo "negative unsigned config value was unexpectedly accepted" >&2
  exit 1
fi

head -c 20 "$fixture/vectors.bin" >"$fixture/truncated.bin"
if "$binary" cpu --vectors "$fixture/truncated.bin" --queries "$fixture/queries.bin" \
    --config "$project_dir/configs/test_ivf.toml" --output "$fixture/invalid.bin"; then
  echo "truncated input was unexpectedly accepted" >&2
  exit 1
fi
echo "CLI integration test passed"
