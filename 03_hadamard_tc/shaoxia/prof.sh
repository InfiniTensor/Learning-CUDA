#!/usr/bin/env bash
# Profile a single (rows, cols) Hadamard kernel launch with Nsight Compute (ncu).
#
# Usage:
#   bash prof.sh                          # defaults: impl 2, fp16, 2048 x 8192
#   bash prof.sh 3 2048 16384 bf16        # positional: IMPL ROWS DIMS [DTYPE]
#   ROWS=1024 DIMS=4096 bash prof.sh      # env overrides work too
#
set -euo pipefail

cd "$(dirname "$0")"

IMPL="${1:-${IMPL:-2}}"
ROWS="${2:-${ROWS:-2048}}"
DIMS="${3:-${DIMS:-8192}}"
DTYPE="${4:-${DTYPE:-fp16}}"

# ncu --kernel-name regex. 'hadamard_kernel' matches impl 1/2/3/4 (scalar/small/
# vec/multi_warp[_chunked]) but not the fht ('fast_hadamard_transform_kernel') or
# hadacore ('hadamard_transform_kernel') baseline.
KERNEL="${KERNEL:-hadamard_kernel_vec}"
SET="${SET:-full}"                              # basic|detailed|full|...
OUT="${OUT:-prof_${IMPL}_${DTYPE}_${ROWS}x${DIMS}}"
NCU_ARGS="${NCU_ARGS:-}"

make bench

# --no-check skips the fp32 reference pass; --no-warmup skips the 1s global warmup;
# --iters 1 --warmup-ms 0 keep the launch count tiny. --kernel-name + --launch-count 1
# then profile only the first matching kernel launch.
sudo ncu \
  --kernel-name "regex:.*(hadamard_kernel_vec|fast_hadamard_transform_kernel).*" \
  --launch-count 8 \
  --set "$SET" \
  -o "$OUT" \
  $NCU_ARGS \
  ./bench --impl "$IMPL" --rows "$ROWS" --dims "$DIMS" --dtype "$DTYPE" \
          --no-check --iters 1 --warmup-ms 10

echo "report: ${OUT}.ncu-rep"
