#!/usr/bin/env bash
#SBATCH --job-name=hadamard-mma
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
    exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
      env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/mma_experiment.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/mma_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/mma_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
COMMON=(-O3 -std=c++17 -arch="sm_$ARCH" -DHADAMARD_TC_VECTOR_SMEM=1 -DHADAMARD_WARPS_PER_BLOCK=0 -I"$ROOT/include" -I"$ROOT/tensor_core/include")
SOURCES=("$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/hadamard_tc.cu" "$ROOT/tensor_core/src/hadamard_mma_experimental.cu")
nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_BRIDGE_TC -DHADAMARD_BRIDGE_MMA \
    "${SOURCES[@]}" "$ROOT/tensor_core/src/python_bridge.cu" -o "$RUN/mma.so" > "$OUT/build.log" 2>&1
nvcc "${COMMON[@]}" "${SOURCES[@]}" "$ROOT/tensor_core/src/mma_smoke.cu" -o "$RUN/mma_smoke" > "$OUT/build_smoke.log" 2>&1
"$RUN/mma_smoke" > "$OUT/smoke.log" 2>&1
for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 "$RUN/mma_smoke" > "$OUT/$tool.log" 2>&1
done
set +e
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" --variant "experiment=$RUN/mma.so" \
    --backends warp,wmma_fast,wmma_split,mma_fast,mma_split --dims 64,128,256 \
    --output-dir "$OUT/validation" > "$OUT/validation.log" 2>&1
echo "validation_rc=$?" > "$OUT/status.txt"
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "experiment=$RUN/mma.so" \
    --backends warp,wmma_fast,wmma_split,mma_fast,mma_split --dims 64,128,256 \
    --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
echo "benchmark_rc=$?" >> "$OUT/status.txt"
set -e
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" > "$OUT/summary.log" 2>&1
cuobjdump --dump-resource-usage "$RUN/mma.so" > "$OUT/resources.txt"
cuobjdump --dump-sass "$RUN/mma.so" > "$RUN/mma.sass"
mkdir -p "$OUT/ncu"
# Dedicated binary mode launches exactly PTX fast then WMMA fast, FP16 d128,
# 16384 tokens, normalized; no PyTorch context/allocator traffic in the process.
for target in ptx_fast wmma_fast; do
    skip=0; [[ "$target" != wmma_fast ]] || skip=1
    stem="$OUT/ncu/${target}_all"
    if ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
        --section Occupancy --cache-control all --clock-control none --launch-skip "$skip" \
        --launch-count 1 -o "$stem" "$RUN/mma_smoke" --profile > "$stem.log" 2>&1; then
        ncu --import "$stem.ncu-rep" --page raw --csv > "${stem}_raw.csv"
    else
        echo "ncu_failed=$target" >> "$OUT/status.txt"
    fi
done
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_mma.py" "$OUT" > "$OUT/mma_summary.log" 2>&1
echo completed=true >> "$OUT/status.txt"
