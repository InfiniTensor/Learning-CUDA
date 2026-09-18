#!/usr/bin/env bash
#SBATCH --job-name=hadamard-library
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
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
: "${REFERENCE_JOB:?set REFERENCE_JOB to the successful CPU reference-build job}"
IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
TARGET=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_${REFERENCE_JOB}/site
# Keep Python, C++ extension ABI and the CUDA toolchain in the same container.
apptainer exec --nv --bind /scratch,/home "$IMAGE" env \
    HADAMARD_STAGE=verify HADAMARD_VERIFY_ONLY=1 HADAMARD_REFERENCE_TARGET="$TARGET" \
    bash "$ROOT/tensor_core/slurm/tc_verify_profile.slurm"
