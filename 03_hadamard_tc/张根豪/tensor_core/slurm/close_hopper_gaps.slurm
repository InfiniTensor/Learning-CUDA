#!/usr/bin/env bash
#SBATCH --job-name=hadamard-hopper-gaps
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
  exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/close_hopper_gaps.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/hopper_gaps_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/hopper_gaps_$SLURM_JOB_ID
mkdir -p "$OUT/ncu" "$RUN/ncu"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; ncu --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
COMMON=(-O3 -std=c++17 -arch=sm_90 -DHADAMARD_TC_VECTOR_SMEM=1 -DHADAMARD_WARPS_PER_BLOCK=0 -I"$ROOT/include" -I"$ROOT/tensor_core/include")
SOURCES=("$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/hadamard_tc.cu" "$ROOT/tensor_core/src/hadamard_mma_experimental.cu")
nvcc "${COMMON[@]}" "${SOURCES[@]}" "$ROOT/tensor_core/src/mma_profile.cu" -o "$RUN/profile" > "$OUT/build_profile.log" 2>&1
nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_BRIDGE_TC -DHADAMARD_BRIDGE_MMA \
  "${SOURCES[@]}" "$ROOT/tensor_core/src/python_bridge.cu" -o "$RUN/mma.so" > "$OUT/build_mma.log" 2>&1
for policy in all none; do
  for dim in 64 128 256; do
    for spec in warp:0 wmma_fast:1 wmma_split:2 ptx_fast:3 ptx_split:4; do
      IFS=: read -r name mode <<< "$spec"
      stem="$OUT/ncu/${name}_d${dim}_${policy}"
      report="$RUN/ncu/${name}_d${dim}_${policy}"
      if timeout --kill-after=5s 90s ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats --section Occupancy \
          --cache-control "$policy" --clock-control none --launch-skip 3 --launch-count 1 \
          -o "$report" "$RUN/profile" "$dim" "$mode" > "$stem.log" 2>&1; then
        ncu --import "$report.ncu-rep" --page raw --csv > "${stem}_raw.csv"
      else
        echo "ncu_failed=$name/$dim/$policy" >> "$OUT/status.txt"
      fi
    done
  done
done
# Same-card PTX/WMMA/warp performance; numerical failures remain explicit.
set +e
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "hopper=$RUN/mma.so" \
  --backends warp,wmma_fast,wmma_split,mma_fast,mma_split --dims 64,128,256 \
  --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
echo "benchmark_rc=$?" >> "$OUT/status.txt"
set -e
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" > "$OUT/summary.log" 2>&1
VARIANTS=()
for spec in 4:0 4:1 8:1; do
  IFS=: read -r f q <<< "$spec"; tag=f${f}_q${q}
  nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_FUSED_WARPS_PER_BLOCK="$f" -DHADAMARD_QUANT_VECTOR_STORE="$q" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
    -o "$RUN/$tag.so" > "$OUT/build_$tag.log" 2>&1
  VARIANTS+=(--variant "$tag=$RUN/$tag.so")
done
# Three independent processes/pools; nine interleaved rounds in each process.
for repeat in 1 2 3; do
  "$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" "${VARIANTS[@]}" --backends fused_warp \
    --dims 512,1024 --tokens 16384,131072 --norms 1 --nodes 16 --rounds 9 \
    --output-dir "$OUT/quant_repeat_$repeat" > "$OUT/quant_repeat_$repeat.log" 2>&1
done
echo completed=true >> "$OUT/status.txt"
