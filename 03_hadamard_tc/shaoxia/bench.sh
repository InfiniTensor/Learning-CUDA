#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

make bench

# ROWS=2048
# DIMS=8192

ROWS=2048
DIMS=2048

# for IMPL in 4; do
#     ./bench --impl "$IMPL" --rows "$ROWS" --dims "$DIMS" --iters 100 --warmup-ms 1
# done

./bench --impl hybrid --rows "$ROWS" --dims "$DIMS"
