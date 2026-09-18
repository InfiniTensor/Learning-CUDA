#!/usr/bin/env bash
#SBATCH --job-name=hadamard-audit
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,h200_tandon,h100_tandon,a100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TC="$ROOT/tensor_core"
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/audit_${SLURM_JOB_ID}
OUT="$TC/results/audit_${SLURM_JOB_ID}"
mkdir -p "$RUN" "$OUT"
for tool_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/nvidia/nsight-compute/*; do
    [[ ! -d "$tool_dir" ]] || export PATH="$tool_dir:$PATH"
done
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
PYTHON="${HADAMARD_TEST_PYTHON:-/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python}"
{
    date --iso-8601=seconds
    nvidia-smi
    nvcc --version
    git -C "$ROOT" diff --stat
} > "$OUT/environment.txt" 2>&1
# Archive precisely the source used by this run; keep build outputs in scratch.
tar -czf "$OUT/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/include tensor_core/Makefile tensor_core/scripts
REFERENCE=cpu
if "$PYTHON" -c 'import fast_hadamard_transform' > "$OUT/library_probe.log" 2>&1; then
    REFERENCE=library
fi
echo "reference=$REFERENCE" > "$OUT/status.txt"
for spec in 4:4:0 2:2:0 8:8:0 4:4:1; do
    IFS=: read -r WARPS TC_WARPS VECTOR <<< "$spec"
    tag=w${WARPS}_tc${TC_WARPS}_v${VECTOR}
    BUILD="$RUN/$tag"
    DEST="$OUT/$tag"
    mkdir -p "$DEST"
    make -C "$TC" OUT="$BUILD" CUDA_ARCH="$ARCH" WARPS="$WARPS" TC_WARPS="$TC_WARPS" TC_VECTOR="$VECTOR" -j4 > "$DEST/build.log" 2>&1
    BIN="$BUILD/hadamard_tc_bench"
    set +e
    "$PYTHON" "$TC/scripts/validate_tc.py" --bin "$BIN" --output-dir "$DEST/validation" --reference "$REFERENCE" > "$DEST/validation.log" 2>&1
    validation_rc=$?
    set -e
    echo "validation_rc=$validation_rc" >> "$DEST/status.txt"
    # Three repetitions preserve variability; no fastest-sample cherry-picking.
    for repeat in 1 2 3; do
        for dtype in fp16 bf16; do
            for dim in 64 128 256 512 1024; do
                "$BIN" --head_dim "$dim" --dtype "$dtype" --normalize true \
                    --csv "$DEST/timing_${repeat}.csv" --warmup 20 --iters 100 \
                    > "$DEST/timing_${repeat}_${dtype}_${dim}.log" 2>&1 || \
                    echo "timing_exit_nonzero repeat=$repeat dtype=$dtype dim=$dim" >> "$DEST/status.txt"
            done
        done
    done
    if command -v compute-sanitizer >/dev/null 2>&1; then
        for sanitizer in memcheck racecheck synccheck; do
            compute-sanitizer --tool "$sanitizer" --error-exitcode 99 "$BIN" \
                --batch 1 --seq 1 --heads 5 --head_dim 64 --dtype bf16 \
                --warmup 0 --iters 1 > "$DEST/${sanitizer}.log" 2>&1 || \
                echo "$sanitizer=failed" >> "$DEST/status.txt"
        done
    fi
done
echo "completed=true" >> "$OUT/status.txt"
