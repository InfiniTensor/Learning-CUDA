#!/usr/bin/env bash
#SBATCH --job-name=hadamard-opt
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
RUN_DIR=/scratch/gz2522/gz2522/tmp/hadamard/runs/optimization_${SLURM_JOB_ID}
FINAL_DIR="$PROJECT_ROOT/results/optimization_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/build" "$FINAL_DIR"

for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin \
    /opt/nvidia/nsight-compute/* /opt/nvidia/nsight-systems/*/bin; do
    [[ -d "$tool_dir" ]] && export PATH="$tool_dir:$PATH"
done
NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"

make -C "$PROJECT_ROOT" OUT="$RUN_DIR/build" CUDA_ARCH="$CUDA_ARCH" -j"$SLURM_CPUS_PER_TASK"
export HADAMARD_ADVANCED_BIN="$RUN_DIR/build/hadamard_advanced_bench"
export HADAMARD_ADVANCED_OUT="$FINAL_DIR"
{
    date --iso-8601=seconds
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.memory --format=csv,noheader
    nvcc --version
} > "$FINAL_DIR/environment.txt" 2>&1
bash "$PROJECT_ROOT/scripts/run_advanced.sh"

# Capture one compact trace containing baseline, optimized, unfused and fused
# phases. The benchmark's kernel symbols keep the implementations separable in
# the exported summary even when hardware counters are unavailable.
if command -v nsys >/dev/null 2>&1; then
    NSYS_STEM="$FINAL_DIR/nsys_advanced_fp16_hd128"
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$NSYS_STEM" \
        "$HADAMARD_ADVANCED_BIN" --batch 1 --seq 512 --heads 32 \
        --head_dim 128 --dtype fp16 --normalize true --warmup 5 --iters 20 || true
    if [[ -f "$NSYS_STEM.nsys-rep" ]]; then
        nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
            --format csv --output "$FINAL_DIR/nsys_advanced_stats" \
            "$NSYS_STEM.nsys-rep" || true
        if [[ -f "$NSYS_STEM.sqlite" ]]; then
            python3 "$PROJECT_ROOT/scripts/summarize_nsys.py" \
                --sqlite "$NSYS_STEM.sqlite" --output-dir "$FINAL_DIR/nsys_summary"
        fi
    fi
fi

# Try one optimized and one fused launch. On clusters where DCGM owns the
# counters this fails harmlessly and leaves explicit logs instead of invented
# occupancy/DRAM numbers.
if command -v ncu >/dev/null 2>&1; then
    mkdir -p "$FINAL_DIR/ncu"
    ncu --set full --replay-mode kernel --clock-control none --launch-skip 1 \
        --launch-count 1 --force-overwrite -o "$FINAL_DIR/ncu/optimized" \
        "$HADAMARD_ADVANCED_BIN" --batch 1 --seq 512 --heads 32 \
        --head_dim 128 --dtype fp16 --normalize true --warmup 0 --iters 1 \
        >"$FINAL_DIR/ncu/optimized.log" 2>&1 || true
    ncu --set full --replay-mode kernel --clock-control none --launch-skip 4 \
        --launch-count 1 --force-overwrite -o "$FINAL_DIR/ncu/fused_int4" \
        "$HADAMARD_ADVANCED_BIN" --batch 1 --seq 512 --heads 32 \
        --head_dim 128 --dtype fp16 --normalize true --warmup 0 --iters 1 \
        >"$FINAL_DIR/ncu/fused_int4.log" 2>&1 || true
fi
