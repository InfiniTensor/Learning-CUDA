#!/bin/bash
# =============================================================================
#  build_musa.sh -- Moore Threads (MUSA) build of the same sources the CUDA
#  build uses.  There is no preprocessing or renaming step: include/fhwt/
#  backend.cuh maps the CUDA spellings onto MUSA's at the include level.
#
#  Usage:   bash scripts/build_musa.sh [target]      target = all | app | tests
#  Env:     MUSA_ROOT (default /usr/local/musa), MUSA_ARCH (default mp_22),
#           BUILD (default build)
#
#  Example (MTT S4000, MUSA 3.1):
#      bash scripts/build_musa.sh all
#      LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH build/test_correctness.exe
# =============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MUSA="${MUSA_ROOT:-/usr/local/musa}"
ARCH="${MUSA_ARCH:-mp_22}"
BUILD="${BUILD:-build}"
TARGET="${1:-all}"

# -mtgpu selects the MT GPU back end; --offload-arch names the device ISA
# (mp_22 = MTT S4000).  -lmusart / -lmusa are the runtime and driver libraries;
# muGraphGetEdges (used by the graph tests) lives in libmusa, not libmusart.
FLAGS="-mtgpu -std=c++17 -O3 -Iinclude -I. --offload-arch=${ARCH} \
       -L${MUSA}/lib -lmusart -lmusa"

mkdir -p "$BUILD"

build() {  # build <output> <sources...>
  out="$1"; shift
  echo "=== mcc $out"
  mcc $FLAGS "$@" -o "$out"
}

COMMON="src/ops.cu"

if [ "$TARGET" = "all" ] || [ "$TARGET" = "app" ]; then
  build "$BUILD/fhwt" src/main.cu $COMMON
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "tests" ]; then
  for t in test_correctness test_fusion test_fp8 test_graph test_gemm; do
    build "$BUILD/$t.exe" "tests/$t.cu" $COMMON
  done
fi
echo "done: $TARGET"
