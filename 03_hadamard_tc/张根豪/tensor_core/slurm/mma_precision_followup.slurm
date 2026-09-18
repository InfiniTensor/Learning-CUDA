#!/usr/bin/env bash
#SBATCH --job-name=hadamard-mma-precision
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
  exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/mma_precision_followup.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/mma_precision_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/mma_precision_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
VARIANTS=()
for precision in 0 1 2; do
  nvcc -O3 -std=c++17 -arch=sm_90 -shared -Xcompiler -fPIC \
    -DHADAMARD_WARPS_PER_BLOCK=0 -DHADAMARD_BRIDGE_MMA -DHADAMARD_MMA_PRECISION="$precision" \
    -I"$ROOT/include" "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" \
    "$ROOT/tensor_core/src/hadamard_mma_experimental.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
    -o "$RUN/p$precision.so" > "$OUT/build_p$precision.log" 2>&1
  VARIANTS+=(--variant "p$precision=$RUN/p$precision.so")
done
set +e
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" "${VARIANTS[@]}" \
  --backends warp,mma_split --dims 64,128,256 --output-dir "$OUT/validation" > "$OUT/validation.log" 2>&1
echo "validation_rc=$?" > "$OUT/status.txt"
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" "${VARIANTS[@]}" \
  --backends warp,mma_split --dims 64,128,256 --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
echo "benchmark_rc=$?" >> "$OUT/status.txt"
set -e
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" > "$OUT/summary.log" 2>&1
echo completed=true >> "$OUT/status.txt"
