#!/usr/bin/env bash
#SBATCH --job-name=hadamard-warp-tune
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
    exec apptainer exec --nv --bind /scratch,/home "$IMAGE" env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/warp_tuning.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/warp_tune_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/warp_tune_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
VARIANTS=()
for spec in 0:0:0 2:2:0 4:4:0 8:8:0 0:2:1 0:4:1 0:8:1; do
    IFS=: read -r WARPS FUSED VECTOR <<< "$spec"
    tag=w${WARPS}_f${FUSED}_q${VECTOR}
    nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="sm_$ARCH" \
        -DHADAMARD_WARPS_PER_BLOCK="$WARPS" -DHADAMARD_FUSED_WARPS_PER_BLOCK="$FUSED" \
        -DHADAMARD_QUANT_VECTOR_STORE="$VECTOR" -I"$ROOT/include" \
        "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
        -o "$RUN/$tag.so" > "$OUT/build_$tag.log" 2>&1
    cuobjdump --dump-sass "$RUN/$tag.so" > "$RUN/$tag.sass"
    VARIANTS+=(--variant "$tag=$RUN/$tag.so")
done
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" "${VARIANTS[@]}" --output-dir "$OUT/validation" > "$OUT/validation.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" "${VARIANTS[@]}" \
    --backends warp,fused_warp --nodes 8 --norms 1 --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" --plot-variant w0_f0_q0 > "$OUT/summary.log" 2>&1
# Representative vector-store candidate; correctness above covers all seven.
make -C "$ROOT/tensor_core" OUT="$RUN/sanitizer" CUDA_ARCH="$ARCH" WARPS=0 FUSED_WARPS=8 QUANT_VECTOR=1 TC_VECTOR=1 -j4 > "$OUT/build_sanitizer.log" 2>&1
for tool in memcheck racecheck synccheck; do
    for dim in 64 512 1024; do
        compute-sanitizer --tool "$tool" --error-exitcode 99 "$RUN/sanitizer/hadamard_tc_bench" \
            --batch 1 --seq 1 --heads 5 --head_dim "$dim" --dtype bf16 --warmup 0 --iters 1 \
            > "$OUT/${tool}_d${dim}.log" 2>&1
    done
done
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_warp_tuning.py" "$OUT" --sass-dir "$RUN" > "$OUT/candidates_summary.log" 2>&1
echo completed=true > "$OUT/status.txt"
