#!/usr/bin/env bash
#SBATCH --job-name=hadamard-guard-quant
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
  exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/guard_and_selective_quant.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/guard_quant_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/guard_quant_$SLURM_JOB_ID
mkdir -p "$OUT" "$RUN"
{ date --iso-8601=seconds; nvidia-smi; nvcc --version; } > "$OUT/environment.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
COMMON=(-O3 -std=c++17 -arch=sm_90 -DHADAMARD_WARPS_PER_BLOCK=0 -I"$ROOT/include")
for spec in control:0:0 selective:2:0 guard:2:3; do
  IFS=: read -r tag quant precision <<< "$spec"
  nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_BRIDGE_MMA \
    -DHADAMARD_QUANT_VECTOR_STORE="$quant" -DHADAMARD_MMA_PRECISION="$precision" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/hadamard_mma_experimental.cu" \
    "$ROOT/tensor_core/src/python_bridge.cu" -o "$RUN/$tag.so" > "$OUT/build_$tag.log" 2>&1
done
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" --variant "control=$RUN/control.so" \
  --variant "selective=$RUN/selective.so" --output-dir "$OUT/quant_validation" > "$OUT/quant_validation.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" --variant "guard=$RUN/guard.so" \
  --backends warp,mma_split --dims 64,128,256 --output-dir "$OUT/guard_validation" > "$OUT/guard_validation.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/stress_guard.py" --library "$RUN/guard.so" --output-dir "$OUT/stress" > "$OUT/stress.log" 2>&1
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 "$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" \
    --variant "guard=$RUN/guard.so" --backends mma_split --dims 64,128,256 \
    --output-dir "$OUT/sanitizer_$tool" > "$OUT/$tool.log" 2>&1
done
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "guard=$RUN/guard.so" \
  --backends warp,mma_split --dims 64,128,256 --output-dir "$OUT/guard_benchmark" > "$OUT/guard_benchmark.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/guard_benchmark" > "$OUT/guard_summary.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "control=$RUN/control.so" \
  --variant "selective=$RUN/selective.so" --backends fused_warp --dims 512,1024 --tokens 16384,131072 --norms 1 \
  --rounds 9 --output-dir "$OUT/quant_benchmark" > "$OUT/quant_benchmark.log" 2>&1
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/quant_benchmark" > "$OUT/quant_summary.log" 2>&1
echo completed=true > "$OUT/status.txt"
