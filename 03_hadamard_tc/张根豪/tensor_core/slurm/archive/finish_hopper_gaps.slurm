#!/usr/bin/env bash
# Historical one-off recovery of job 17291971, not a general profiling entry.
#SBATCH --job-name=hadamard-hopper-finish
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodelist=gh108
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err
set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ "${HADAMARD_CONTAINER:-0}" != 1 ]]; then
  exec apptainer exec --nv --bind /scratch,/home /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    env HADAMARD_CONTAINER=1 bash "$ROOT/tensor_core/slurm/archive/finish_hopper_gaps.slurm"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/scratch/gz2522/gz2522/tmp/hadamard/runs/reference_17251238/site:${PYTHONPATH:-}
PYTHON=/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python
OUT="$ROOT/tensor_core/results/hopper_gaps_$SLURM_JOB_ID"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/hopper_gaps_$SLURM_JOB_ID
ORIGIN="$ROOT/tensor_core/results/hopper_gaps_17291971"
ORIGIN_RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/hopper_gaps_17291971
mkdir -p "$OUT/ncu" "$RUN"
{ date --iso-8601=seconds; nvidia-smi; nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv; nvcc --version; } > "$OUT/environment.txt"
cp "$ORIGIN"/ncu/*_raw.csv "$ORIGIN"/ncu/*.log "$OUT/ncu/"
cp "$ORIGIN/environment.txt" "$OUT/ncu_environment.txt"
echo 'ncu_origin=17291971; performance=current job; same H200 node; UUID checked separately' > "$OUT/provenance.txt"
tar -czf "$RUN/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/scripts
set +e
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" --variant "hopper=$ORIGIN_RUN/mma.so" \
  --backends warp,wmma_fast,wmma_split,mma_fast,mma_split --dims 64,128,256 \
  --output-dir "$OUT/benchmark" > "$OUT/benchmark.log" 2>&1
echo "benchmark_rc=$?" > "$OUT/status.txt"
set -e
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/benchmark" > "$OUT/summary.log" 2>&1
COMMON=(-O3 -std=c++17 -arch=sm_90 -DHADAMARD_WARPS_PER_BLOCK=0 -I"$ROOT/include")
VARIANTS=()
for spec in 4:0 4:1 8:1; do
  IFS=: read -r f q <<< "$spec"; tag=f${f}_q${q}
  nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_FUSED_WARPS_PER_BLOCK="$f" -DHADAMARD_QUANT_VECTOR_STORE="$q" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/python_bridge.cu" \
    -o "$RUN/$tag.so" > "$OUT/build_$tag.log" 2>&1
  VARIANTS+=(--variant "$tag=$RUN/$tag.so")
done
for repeat in 1 2 3; do
  "$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" "${VARIANTS[@]}" --backends fused_warp \
    --dims 512,1024 --tokens 16384,131072 --norms 1 --nodes 16 --rounds 9 \
    --output-dir "$OUT/quant_repeat_$repeat" > "$OUT/quant_repeat_$repeat.log" 2>&1
done
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_hopper.py" "$OUT" --dram-peak-gbs 4800 --fp32-peak-tflops 67 > "$OUT/hopper_summary.log" 2>&1
VARIANTS=()
for precision in 0 1 2; do
  nvcc "${COMMON[@]}" -shared -Xcompiler -fPIC -DHADAMARD_BRIDGE_MMA -DHADAMARD_MMA_PRECISION="$precision" \
    "$ROOT/src/hadamard.cu" "$ROOT/src/hadamard_warp.cu" "$ROOT/tensor_core/src/hadamard_mma_experimental.cu" \
    "$ROOT/tensor_core/src/python_bridge.cu" -o "$RUN/p$precision.so" > "$OUT/build_p$precision.log" 2>&1
  VARIANTS+=(--variant "p$precision=$RUN/p$precision.so")
done
set +e
"$PYTHON" "$ROOT/tensor_core/scripts/validate_bridge.py" "${VARIANTS[@]}" --backends warp,mma_split \
  --dims 64,128,256 --output-dir "$OUT/precision_validation" > "$OUT/precision_validation.log" 2>&1
echo "precision_validation_rc=$?" >> "$OUT/status.txt"
"$PYTHON" "$ROOT/tensor_core/scripts/benchmark_library.py" "${VARIANTS[@]}" --backends warp,mma_split \
  --dims 64,128,256 --output-dir "$OUT/precision_benchmark" > "$OUT/precision_benchmark.log" 2>&1
echo "precision_benchmark_rc=$?" >> "$OUT/status.txt"
set -e
"$PYTHON" "$ROOT/tensor_core/scripts/summarize_library.py" "$OUT/precision_benchmark" > "$OUT/precision_summary.log" 2>&1
echo completed=true >> "$OUT/status.txt"
