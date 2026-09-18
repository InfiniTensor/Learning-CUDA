#!/usr/bin/env bash
# Retry hardware-counter collection on an explicitly selected GPU partition.
# Submit examples:
#   sbatch --partition=l40s_public slurm/ncu_retry.slurm
#   sbatch --partition=a100_tandon slurm/ncu_retry.slurm
#   sbatch --partition=h100_tandon slurm/ncu_retry.slurm
#   sbatch --partition=h200_tandon slurm/ncu_retry.slurm
#SBATCH --job-name=hadamard-ncu
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_DIR=/scratch/gz2522/gz2522/tmp/hadamard/runs/ncu_retry_${SLURM_JOB_ID}
FINAL_DIR="$PROJECT_ROOT/results/ncu_retry_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/build" "$FINAL_DIR/raw"

for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin \
    /opt/nvidia/nsight-compute/*; do
    [[ -d "$tool_dir" ]] && export PATH="$tool_dir:$PATH"
done

NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
make -C "$PROJECT_ROOT" OUT="$RUN_DIR/build" CUDA_ARCH="$CUDA_ARCH" \
    -j"$SLURM_CPUS_PER_TASK"

BIN="$RUN_DIR/build/hadamard_advanced_bench"
COMMON=(--batch 1 --seq 512 --heads 32 --head_dim 128 --dtype fp16 \
        --normalize true --warmup 0 --iters 1)
{
    date --iso-8601=seconds
    echo "host=$(hostname)"
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version \
        --format=csv,noheader
    nvcc --version
    ncu --version
} > "$FINAL_DIR/environment.txt" 2>&1

run_profile() {
    local name="$1"
    local skip="$2"
    local report="$FINAL_DIR/$name"
    local log="$FINAL_DIR/$name.log"
    if ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
        --section Occupancy --section WarpStateStats --replay-mode kernel \
        --clock-control none --launch-skip "$skip" --launch-count 1 \
        --force-overwrite -o "$report" "$BIN" "${COMMON[@]}" \
        >"$log" 2>&1; then
        ncu --import "$report.ncu-rep" --csv --page raw \
            > "$FINAL_DIR/raw/$name.csv" 2>>"$log"
        echo "$name=success" | tee -a "$FINAL_DIR/status.txt"
    else
        echo "$name=failed" | tee -a "$FINAL_DIR/status.txt"
    fi
}

# Validation launch order in advanced_bench:
# 0 baseline, 1 optimized, 2 optimized (unfused), 3 quantize, 4 fused.
run_profile optimized 1
run_profile fused_int4 4

if [[ -f "$FINAL_DIR/raw/optimized.csv" && \
      -f "$FINAL_DIR/raw/fused_int4.csv" ]]; then
    python3 "$PROJECT_ROOT/scripts/summarize_ncu_retry.py" \
        --input-dir "$FINAL_DIR/raw" --output "$FINAL_DIR/ncu_summary.csv"
    if python3 -c 'import matplotlib' >/dev/null 2>&1; then
        python3 "$PROJECT_ROOT/scripts/plot_ncu_retry.py" \
            --input "$FINAL_DIR/ncu_summary.csv" \
            --output "$FINAL_DIR/ncu_hardware_metrics.png"
    fi
fi
