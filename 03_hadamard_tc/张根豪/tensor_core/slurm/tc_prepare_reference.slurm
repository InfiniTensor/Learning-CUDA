#!/usr/bin/env bash
#SBATCH --job-name=hadamard-ref-build
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

# Build the external reference without holding a GPU. CUDA 12.8 matches the
# PyTorch CUDA major; do not bypass PyTorch's CUDA compatibility check.
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_${SLURM_JOB_ID}
OUT="$ROOT/tensor_core/results/reference_${SLURM_JOB_ID}"
IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
mkdir -p "$RUN/site" "$OUT"
# PyPI 1.1.0's source tarball omits csrc, so build the complete tagged upstream
# checkout instead. Record its exact commit and leave the checkout unmodified.
git clone --branch v1.1.0 --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/Dao-AILab/fast-hadamard-transform.git "$RUN/source" \
    > "$OUT/checkout.log" 2>&1
git -C "$RUN/source" rev-parse HEAD > "$OUT/source_commit.txt"
apptainer exec --cleanenv --bind /scratch,/home "$IMAGE" env \
    CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0' \
    "$PYTHON" -m pip install --no-deps --no-build-isolation --target "$RUN/site" \
    "$RUN/source" > "$OUT/build.log" 2>&1
apptainer exec --cleanenv --bind /scratch,/home "$IMAGE" env \
    CUDA_HOME=/usr/local/cuda PYTHONPATH="$RUN/site" \
    "$PYTHON" -c 'import sys, torch, fast_hadamard_transform; print(sys.version); print(torch.__version__); print(fast_hadamard_transform.__file__)' \
    > "$OUT/import_check.txt" 2>&1
echo "reference_target=$RUN/site" > "$OUT/status.txt"
