#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make fuse_bench

ROWS="${ROWS:-32,2048}"
DIMS="${DIMS:-16384}"
DTYPE="${DTYPE:-all}"
QUANT="${QUANT:-all}"
SCHEME="${SCHEME:-all}"
POLICY="${POLICY:-auto}"
ITERS="${ITERS:-100}"
WARMUP_MS="${WARMUP_MS:-20}"

./fuse_bench \
  --rows "$ROWS" \
  --dims "$DIMS" \
  --dtype "$DTYPE" \
  --quant "$QUANT" \
  --scheme "$SCHEME" \
  --policy "$POLICY" \
  --iters "$ITERS" \
  --warmup-ms "$WARMUP_MS"
