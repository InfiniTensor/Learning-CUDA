#!/bin/bash
# =============================================================================
#  run_experiments_musa.sh -- the MUSA (Moore Threads) experiment sweep.
#
#  Mirrors `make bench` for the CUDA build; writes into data/musa/ so that the
#  two platforms' artefacts never overwrite each other.  Needs build_musa.sh to
#  have been run first (or pass --build).
#
#  Usage:  bash scripts/run_experiments_musa.sh [--build]
# =============================================================================
set -e
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
B=./build/fhwt
OUT=data/musa
mkdir -p "$OUT"

if [ "$1" = "--build" ]; then bash scripts/build_musa.sh all; fi

# 0. device properties + the measured bandwidth roof
$B --mode info | tee "$OUT/info.txt"
# The driver's theoretical DRAM figure is *below* what a copy sustains on this
# platform, so the roof is measured rather than assumed (see fhwt/probe.cuh).
$B --mode roof | tee "$OUT/roof.txt"

# 1. launch-geometry sweep
$B --mode tune --rows 65536 --dtype f16 --dims 64,128,256,512,1024 \
   --iters 50 --warmup 10 --reps 3 --csv "$OUT/tune.csv"

# 2. register / shuffle kernel matrix (dim <= 1024)
$B --mode matrix --kernel reg --iters 50 --warmup 10 --reps 3 --csv "$OUT/kernels_reg.csv"

# 3. shared-memory kernel matrix (large head dimensions)
$B --mode matrix --kernel smem --dims 64,128,256,512,1024,2048,4096,8192,16384 \
   --batch 2048,8192,32768,131072 --iters 50 --warmup 10 --reps 3 --csv "$OUT/kernels_smem.csv"

# 4. in-place vs out-of-place (the L2 capacity effect)
$B --mode inplace --dims 128,256,512,1024 --dtypes f16,bf16 \
   --batch 8192,16384,32768,65536,131072,262144 \
   --iters 100 --warmup 10 --reps 5 --csv "$OUT/inplace.csv"

# 5. fused quantisation
for q in fp8 int4; do
  for d in 64 128 256 512 1024; do
    $B --mode quant --quant $q --dim $d --rows 262144 --dtype f16 --iters 50 --warmup 10 \
       > "$OUT/quant_${q}_${d}.log"
  done
done

# 6. graph experiments (the sweeps live in tools/, so the CSV format matches the
#    CUDA run; --runs is small here because a single MUSA sweep is much slower)
python3 tools/bench_graph.py --bin $B --out $OUT
python3 tools/bench_loop.py  --bin $B --out $OUT --reps 3 --runs 5
python3 tools/bench_roots.py --bin $B --out $OUT --reps 3 --runs 5
python3 tools/bench_gemm.py  --bin $B --out $OUT --reps 3 --runs 5
python3 tools/bench_group.py --bin $B --out $OUT --reps 3 --runs 3
# bench_batch.py sweeps 5 shapes x K in {1,2,5,10,20} x 8 rounds, which is hours
# on this part; the K sweep below covers the report's 7.14 claim for one shape.
for K in 1 2 5 10 20; do
  $B --mode loop --rows 512 --dims 64,128,256 --layers 8 --tensors 6 --group-entry \
     --passes-per-graph $K --iters 20 --dtype f16 --reps 3 >> "$OUT/passes.log"
done

# 7. CPU baseline
$B --mode cpu --dim 256 --rows 65536 --threads 8 > "$OUT/cpu.log"

# 8. correctness / regression suite
for t in test_correctness test_fusion test_fp8 test_graph test_gemm; do
  ./build/$t.exe | tail -1
done > "$OUT/tests.log"
cat "$OUT/tests.log"
echo "MUSA_SWEEP_DONE"
