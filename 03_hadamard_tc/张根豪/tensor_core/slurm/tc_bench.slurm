#!/usr/bin/env bash
#SBATCH --job-name=hadamard-tc
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,h100_tandon,h200_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

# Tensor Core 对照实验：构建 -> 正确性与性能全矩阵 -> nsys kernel timeline
# -> ncu 硬件计数器（若节点未被 DCGM 占用）。所有原始产物落在
# results/tc_<job-id>/，不覆盖主项目已有结果。

set -euo pipefail

PROJECT_ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TC_ROOT="$PROJECT_ROOT/tensor_core"
RUN_DIR=/scratch/gz2522/gz2522/tmp/hadamard/runs/tc_${SLURM_JOB_ID}
FINAL_DIR="$TC_ROOT/results/tc_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/build" "$FINAL_DIR"

for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin \
    /opt/nvidia/nsight-compute/* /opt/nvidia/nsight-systems/*/bin; do
    [[ -d "$tool_dir" ]] && export PATH="$tool_dir:$PATH"
done
NVCC_PATH="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"

{
    date --iso-8601=seconds
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.memory --format=csv,noheader
    nvcc --version
    echo "CUDA_ARCH=$CUDA_ARCH"
} > "$FINAL_DIR/environment.txt" 2>&1

make -C "$TC_ROOT" OUT="$RUN_DIR/build" CUDA_ARCH="$CUDA_ARCH" -j"$SLURM_CPUS_PER_TASK"
BIN="$RUN_DIR/build/hadamard_tc_bench"

# 主项目的 9.1/9.2 A/B 也重跑一次，保证 TC 数据与非 TC 数据来自同一块 GPU、
# 同一次调度，不跨节点混算。
make -C "$PROJECT_ROOT" OUT="$RUN_DIR/build_main" CUDA_ARCH="$CUDA_ARCH" -j"$SLURM_CPUS_PER_TASK"

export HADAMARD_TC_BIN="$BIN"
export HADAMARD_TC_OUT="$FINAL_DIR"
# 扫描里若有个别配置未通过，仍然继续采 profiler 数据；退出码单独记录，
# 不让 pipefail 把后面的 nsys/ncu 阶段一起吃掉。
set +e
bash "$TC_ROOT/scripts/run_tc.sh" 2>&1 | tee "$FINAL_DIR/sweep.log"
SWEEP_RC=${PIPESTATUS[0]}
set -e
echo "sweep_exit_code=$SWEEP_RC" > "$FINAL_DIR/status.txt"

# Nsight Systems：一条 trace 同时包含 baseline/optimized/warp/tc_fast/tc_split
# 以及四条融合路径，kernel 符号名足以在导出的汇总里区分。
if command -v nsys >/dev/null 2>&1; then
    NSYS_STEM="$FINAL_DIR/nsys_tc_fp16_hd128"
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$NSYS_STEM" \
        "$BIN" --batch 1 --seq 512 --heads 32 --head_dim 128 --dtype fp16 \
        --normalize true --warmup 5 --iters 20 --ref_tokens 1024 || true
    if [[ -f "$NSYS_STEM.nsys-rep" ]]; then
        nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
            --format csv --output "$FINAL_DIR/nsys_tc_stats" \
            "$NSYS_STEM.nsys-rep" || true
    fi
fi

# Nsight Compute：分别采 warp FHT 与两个 Tensor Core kernel。节点被 DCGM 占用
# 时这里会失败，脚本保留原始日志而不是编造硬件计数器。
if command -v ncu >/dev/null 2>&1; then
    mkdir -p "$FINAL_DIR/ncu"
    for target in warp tc_fast tc_split; do
        case "$target" in
            warp)     skip=2 ;;
            tc_fast)  skip=3 ;;
            tc_split) skip=4 ;;
        esac
        ncu --set full --replay-mode kernel --clock-control none \
            --launch-skip "$skip" --launch-count 1 --force-overwrite \
            -o "$FINAL_DIR/ncu/$target" \
            "$BIN" --batch 1 --seq 512 --heads 32 --head_dim 128 --dtype fp16 \
            --normalize true --warmup 0 --iters 1 --ref_tokens 256 \
            >"$FINAL_DIR/ncu/$target.log" 2>&1 || true
    done
fi

# 计算节点的 python3 不一定装了 matplotlib；脚本会自动降级为只输出表格，
# 图表可在登录节点用同一条命令补齐。
python3 "$TC_ROOT/scripts/summarize_tc.py" --csv "$FINAL_DIR/tc_results.csv" \
    --output-dir "$FINAL_DIR" || true

echo "[tc] artifacts in $FINAL_DIR (sweep_exit_code=$SWEEP_RC)"
exit "$SWEEP_RC"
