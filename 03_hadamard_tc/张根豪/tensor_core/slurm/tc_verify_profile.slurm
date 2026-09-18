#!/usr/bin/env bash
#SBATCH --job-name=hadamard-verify
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public,a100_tandon,h100_tandon,h200_tandon
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
STAGE="${HADAMARD_STAGE:-verify}"
case "$STAGE" in verify|ncu) ;; *) echo "invalid HADAMARD_STAGE=$STAGE" >&2; exit 2 ;; esac
RUN=/scratch/gz2522/gz2522/tmp/hadamard/runs/${STAGE}_${SLURM_JOB_ID}
OUT="$TC/results/${STAGE}_${SLURM_JOB_ID}"
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
    if [[ "$STAGE" == verify ]]; then
        "$PYTHON" -c 'import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())'
    fi
} > "$OUT/environment.txt" 2>&1
tar -czf "$OUT/source_snapshot.tar.gz" -C "$ROOT" src include tensor_core/src tensor_core/include tensor_core/Makefile tensor_core/scripts
# Read the existing CUDA PyTorch environment, but install the reference library
# only into this job's isolated target. Never modify another project's venv.
export MAX_JOBS=4
REFERENCE=not_used
install_rc=not_run
if [[ "$STAGE" == verify ]]; then
if [[ -n "${HADAMARD_REFERENCE_TARGET:-}" ]]; then
    export PYTHONPATH="$HADAMARD_REFERENCE_TARGET:${PYTHONPATH:-}"
    "$PYTHON" -c 'import fast_hadamard_transform' > "$OUT/library_probe.log" 2>&1
    REFERENCE=library
    install_rc=prebuilt
else
set +e
timeout --kill-after=5s 300s "$PYTHON" -m pip install --no-deps --no-build-isolation --target "$RUN/library" \
    fast_hadamard_transform==1.1.0 > "$OUT/library_install.log" 2>&1
install_rc=$?
set -e
export PYTHONPATH="$RUN/library:${PYTHONPATH:-}"
REFERENCE=cpu
if "$PYTHON" -c 'import fast_hadamard_transform' > "$OUT/library_probe.log" 2>&1; then
    REFERENCE=library
fi
fi
fi
echo "stage=$STAGE reference=$REFERENCE install_rc=$install_rc" > "$OUT/status.txt"
for spec in ${HADAMARD_SPECS:-4:4:0 8:8:0 4:4:1 0:4:1 0:4:2}; do
    IFS=: read -r WARPS TC_WARPS VECTOR <<< "$spec"
    tag=w${WARPS}_tc${TC_WARPS}_v${VECTOR}
    DEST="$OUT/$tag"
    mkdir -p "$DEST/ncu"
    make -C "$TC" OUT="$RUN/$tag" CUDA_ARCH="$ARCH" WARPS="$WARPS" TC_WARPS="$TC_WARPS" TC_VECTOR="$VECTOR" -j4 > "$DEST/build.log" 2>&1
    BIN="$RUN/$tag/hadamard_tc_bench"
    if [[ "$STAGE" == verify ]]; then
    set +e
    "$PYTHON" "$TC/scripts/validate_tc.py" --bin "$BIN" --output-dir "$DEST/validation" \
        --reference "$REFERENCE" > "$DEST/validation.log" 2>&1
    validation_rc=$?
    set -e
    echo "validation_rc=$validation_rc" > "$DEST/status.txt"
    if [[ "${HADAMARD_VERIFY_ONLY:-0}" == 1 ]]; then continue; fi
    # Keep measurements from this GPU separate from the earlier H200 sweep.
    for repeat in 1 2 3; do
        for dtype in fp16 bf16; do
            for dim in 64 128 256; do
                "$BIN" --head_dim "$dim" --dtype "$dtype" --normalize true \
                    --warmup 20 --iters 100 --csv "$DEST/timing_${repeat}.csv" \
                    > "$DEST/timing_${repeat}_${dtype}_${dim}.log" 2>&1 || \
                    echo "timing_nonzero=$repeat/$dtype/$dim" >> "$DEST/status.txt"
            done
        done
    done
    for sanitizer in memcheck racecheck synccheck; do
        for dim in 64 128 256 512 1024; do
            compute-sanitizer --tool "$sanitizer" --error-exitcode 99 "$BIN" \
                --batch 1 --seq 1 --heads 5 --head_dim "$dim" --dtype bf16 \
                --warmup 0 --iters 1 > "$DEST/${sanitizer}_d${dim}.log" 2>&1 || \
                echo "sanitizer_nonzero=$sanitizer/$dim" >> "$DEST/status.txt"
        done
    done
    continue
    fi
    # all: profiler flushes caches; none: cache state may carry over. Neither
    # mode is labelled cold/warm without checking the measured traffic.
    for policy in all none; do
        for target in warp tc_fast tc_split warp_d64; do
            PROFILE_DIM=128
            case "$target" in warp) skip=2 ;; tc_fast) skip=3 ;; tc_split) skip=4 ;;
                warp_d64) skip=2; PROFILE_DIM=64 ;; esac
            stem="$DEST/ncu/${target}_${policy}"
            ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
                --section WarpStateStats --section SchedulerStats \
                --cache-control "$policy" --replay-mode kernel --clock-control none \
                --launch-skip "$skip" --launch-count 1 --force-overwrite -o "$stem" \
                "$BIN" --batch 1 --seq 512 --heads 32 --head_dim "$PROFILE_DIM" --dtype fp16 \
                --normalize true --warmup 0 --iters 1 --ref_tokens 256 \
                > "$stem.log" 2>&1 || echo "ncu_failed=$target/$policy" >> "$DEST/status.txt"
            if [[ -f "$stem.ncu-rep" ]]; then
                ncu --import "$stem.ncu-rep" --csv --page raw > "${stem}_raw.csv" 2>> "$DEST/status.txt"
            fi
        done
    done
done
echo "completed=true" >> "$OUT/status.txt"
if [[ "$STAGE" == verify && "${HADAMARD_VERIFY_ONLY:-0}" != 1 ]]; then
    "$PYTHON" "$TC/scripts/summarize_audit.py" --timing-dir "$OUT" --verify-dir "$OUT" \
        --output-dir "$OUT/summary" > "$OUT/summary_generation.log" 2>&1
fi
