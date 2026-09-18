#!/usr/bin/env bash
#SBATCH --job-name=hadamard-libperf
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
    exec apptainer exec --nv --bind /scratch,/home "$IMAGE" env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/library_performance.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/libperf_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/libperf_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="sm_$ARCH" \
    -DHADAMARD_WARPS_PER_BLOCK=0 -I"$ROOT/include" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
    -o "$RUN/libhadamard.so" > "$OUT/build.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "tuned=$RUN/libhadamard.so" \
    --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" > "$OUT/summary.log" 2>&1
echo completed=true > "$OUT/status.txt"
