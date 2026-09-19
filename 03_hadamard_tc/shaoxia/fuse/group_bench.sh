#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make fuse_group_bench

ROWS="${ROWS:-2048}"
COLS="${COLS:-8192}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DTYPE="${DTYPE:-all}"
QUANT="${QUANT:-all}"
SCHEME="${SCHEME:-all}"
ITERS="${ITERS:-100}"
WARMUP_MS="${WARMUP_MS:-20}"

./fuse_group_bench \
  --rows "$ROWS" \
  --cols "$COLS" \
  --group-size "$GROUP_SIZE" \
  --dtype "$DTYPE" \
  --quant "$QUANT" \
  --scheme "$SCHEME" \
  --iters "$ITERS" \
  --warmup-ms "$WARMUP_MS"
