#!/usr/bin/env bash
#SBATCH --job-name=hadamard-compile
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
    exec apptainer exec --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
      env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/compile_experiments.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
OUT="$ROOT/tensor_core/results/compile_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/compile_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
COMMON=(-O3 -std=c++17 -arch=sm_90 -I"$ROOT/include" -I"$ROOT/tensor_core/include")
nvcc "${COMMON[@]}" -DHADAMARD_TC_VECTOR_SMEM=1 -c "$ROOT/tensor_core/src/tc_trace.cu" -o "$RUN/trace.o" > "$OUT/trace.log" 2>&1
nvcc "${COMMON[@]}" -c "$ROOT/tensor_core/src/hadamard_mma_experimental.cu" -o "$RUN/mma.o" > "$OUT/mma.log" 2>&1
nvcc "${COMMON[@]}" -c "$ROOT/tensor_core/src/mma_smoke.cu" -o "$RUN/smoke.o" > "$OUT/smoke.log" 2>&1
nvcc "${COMMON[@]}" -DHADAMARD_BRIDGE_TC -DHADAMARD_BRIDGE_MMA -c "$ROOT/tensor_core/src/python_bridge.cu" -o "$RUN/bridge.o" > "$OUT/bridge.log" 2>&1
echo compiled=true > "$OUT/status.txt"
