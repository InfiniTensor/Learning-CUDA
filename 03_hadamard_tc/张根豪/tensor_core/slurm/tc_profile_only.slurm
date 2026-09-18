#!/usr/bin/env bash
#SBATCH --job-name=hadamard-counters
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

# Independent of the correctness queue: no torch/import/install prerequisite.
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HADAMARD_STAGE=ncu
export HADAMARD_SPECS="4:4:0 8:8:0 4:4:1 0:4:1 0:4:2"
bash "$ROOT/tensor_core/slurm/tc_verify_profile.slurm"
