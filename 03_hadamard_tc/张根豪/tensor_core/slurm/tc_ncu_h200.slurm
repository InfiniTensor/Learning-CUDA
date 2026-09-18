#!/usr/bin/env bash
#SBATCH --job-name=hadamard-tc-ncu
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

# 主 sweep 所在的 L40S 分区上 performance counter 被 DCGM 占用。本任务只做一件
# 事：在 Hopper 节点上给 warp / tc_fast / tc_split 三个 kernel 采真实硬件计数器。
# 与主项目 9.1/9.2 的 H200 NCU 重试口径一致：计数器用于 kernel 结构诊断，
# 不与 L40S 的端到端计时混算。

set -euo pipefail

PROJECT_ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TC_ROOT="$PROJECT_ROOT/tensor_core"
RUN_DIR=/scratch/gz2522/gz2522/tmp/hadamard/runs/tc_ncu_${SLURM_JOB_ID}
FINAL_DIR="$TC_ROOT/results/tc_ncu_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" "$FINAL_DIR/ncu"

for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin \
    /opt/nvidia/nsight-compute/* ; do
    [[ -d "$tool_dir" ]] && export PATH="$tool_dir:$PATH"
done
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"

{
    date --iso-8601=seconds
    hostname
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    nvcc --version | tail -2
    ncu --version | head -3
} > "$FINAL_DIR/environment.txt" 2>&1

make -C "$TC_ROOT" OUT="$RUN_DIR" CUDA_ARCH="$CUDA_ARCH" -j"$SLURM_CPUS_PER_TASK"
BIN="$RUN_DIR/hadamard_tc_bench"

# --launch-skip 依赖 tc_bench 正确性阶段固定的 kernel 启动顺序：
#   0 baseline, 1 optimized, 2 warp, 3 tc_fast, 4 tc_split
SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis \
--section Occupancy --section WarpStateStats --section SchedulerStats"
status=0
for target in warp tc_fast tc_split; do
    case "$target" in
        warp)     skip=2 ;;
        tc_fast)  skip=3 ;;
        tc_split) skip=4 ;;
    esac
    ncu $SECTIONS --replay-mode kernel --clock-control none \
        --launch-skip "$skip" --launch-count 1 --force-overwrite \
        --csv --page raw -o "$FINAL_DIR/ncu/$target" \
        "$BIN" --batch 1 --seq 512 --heads 32 --head_dim 128 --dtype fp16 \
        --normalize true --warmup 0 --iters 1 --ref_tokens 256 \
        >"$FINAL_DIR/ncu/$target.log" 2>&1 || status=1
done

# 有报告就导出 raw CSV，供 summarize 脚本提取关键指标。
for target in warp tc_fast tc_split; do
    rep="$FINAL_DIR/ncu/$target.ncu-rep"
    [[ -f "$rep" ]] && ncu --import "$rep" --csv --page raw \
        > "$FINAL_DIR/ncu/${target}_raw.csv" 2>/dev/null || true
done

# 从 600 多列的 raw CSV 里抽出用于结构诊断的那几项。
python3 "$TC_ROOT/scripts/summarize_tc_ncu.py" --ncu-dir "$FINAL_DIR/ncu" \
    --output-dir "$FINAL_DIR" || true

echo "ncu_exit_status=$status" > "$FINAL_DIR/status.txt"
echo "[tc-ncu] artifacts in $FINAL_DIR (status=$status)"
