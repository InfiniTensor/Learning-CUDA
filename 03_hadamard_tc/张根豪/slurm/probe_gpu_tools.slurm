#!/usr/bin/env bash
#SBATCH --job-name=hadamard-probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,l40s_public,h100_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

set -euo pipefail

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.memory --format=csv,noheader

for tool in nvcc ncu nsys cmake python3; do
    path="$(command -v "$tool" || true)"
    echo "$tool=${path:-missing}"
    if [[ -n "$path" ]]; then
        "$tool" --version 2>&1 | head -n 8 || true
    fi
done

echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

if command -v ncu >/dev/null 2>&1; then
    echo "ncu sections containing roofline:"
    ncu --list-sections 2>&1 | grep -i roofline || true
fi

echo "candidate CUDA/Nsight binaries:"
find /usr/local /opt /share/apps -maxdepth 6 -type f \
    \( -name nvcc -o -name ncu -o -name nsys \) -executable 2>/dev/null | sort | head -n 100 || true

echo "CUDA-related modules:"
module -t avail 2>&1 | grep -Ei 'cuda|nsight|nvidia' | head -n 100 || true
