#!/usr/bin/env bash
#SBATCH --job-name=hadamard-gap-builds
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
  exec apptainer exec --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/check_gap_builds.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
OUT="$ROOT/tensor_core/results/gap_builds_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/gap_builds_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
cmake -S "$ROOT" -B "$RUN/cmake" -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DHADAMARD_WARPS_PER_BLOCK=0 -DHADAMARD_FUSED_WARPS_PER_BLOCK=0 -DHADAMARD_QUANT_VECTOR_STORE=2 \
  > "$OUT/cmake_configure.log" 2>&1
cmake --build "$RUN/cmake" -j4 > "$OUT/cmake_build.log" 2>&1
make -C "$ROOT" -j4 CUDA_ARCH=90 WARPS=0 QUANT_VECTOR=2 OUT="$RUN/make" > "$OUT/make_build.log" 2>&1
make -C "$ROOT/tensor_core" -j2 mma-experiment CUDA_ARCH=90 WARPS=0 QUANT_VECTOR=2 MMA_PRECISION=3 \
  OUT="$RUN/mma" > "$OUT/mma_build.log" 2>&1
echo completed=true > "$OUT/status.txt"
