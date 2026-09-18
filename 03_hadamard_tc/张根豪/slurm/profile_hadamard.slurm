#!/usr/bin/env bash
#SBATCH --job-name=hadamard-prof
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,h100_tandon,h200_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_DIR=/scratch/gz2522/gz2522/tmp/hadamard/runs/profile_${SLURM_JOB_ID}
FINAL_DIR="$PROJECT_ROOT/results/profile_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/build" "$FINAL_DIR"

# GPU nodes expose CUDA/Nsight in one of these standard locations, but do not
# necessarily load them on the login node.
for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/nvidia/nsight-compute/* /opt/nvidia/nsight-systems/*/bin; do
    if [[ -d "$tool_dir" ]]; then
        export PATH="$tool_dir:$PATH"
    fi
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc was not found on $(hostname)" >&2
    exit 2
fi

NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
make -C "$PROJECT_ROOT" OUT="$RUN_DIR/build" CUDA_ARCH="$CUDA_ARCH" -j"$SLURM_CPUS_PER_TASK"

export HADAMARD_BIN="$RUN_DIR/build/hadamard_bench"
export HADAMARD_PROFILE_OUT="$FINAL_DIR"
bash "$PROJECT_ROOT/scripts/profile_matrix.sh"
