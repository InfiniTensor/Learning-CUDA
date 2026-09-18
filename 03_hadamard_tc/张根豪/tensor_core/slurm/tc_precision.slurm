#!/usr/bin/env bash
#SBATCH --job-name=hadamard-precision
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
    exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
      env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/tc_precision.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/precision_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/precision_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="sm_$ARCH" -DHADAMARD_BRIDGE_TC \
    -DHADAMARD_TC_VECTOR_SMEM=1 -I"$ROOT/include" -I"$ROOT/tensor_core/include" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" \
    "$ROOT/tensor_core/src/tc_trace.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
    -o "$RUN/trace.so" > "$OUT/build.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/diagnose_precision.py" --library "$RUN/trace.so" \
    --validation-dir "$ROOT/tensor_core/results/verify_17251286/w0_tc4_v1/validation" \
    --output-dir "$OUT/diagnosis" > "$OUT/diagnosis.log" 2>&1
echo completed=true > "$OUT/status.txt"
