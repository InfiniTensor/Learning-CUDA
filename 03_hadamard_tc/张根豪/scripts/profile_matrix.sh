#!/usr/bin/env bash
# Collect timing, Nsight Systems, and Nsight Compute data for the core matrix.
# Run this inside a one-GPU allocation; the Slurm wrapper is in slurm/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HADAMARD_BIN:-$ROOT/build/hadamard_bench}"
OUT="${HADAMARD_PROFILE_OUT:-$ROOT/results/profile_run}"
TOKENS=131072
BATCH=4
SEQ=1024
HEADS=32
NCU_FAILURES=0

mkdir -p "$OUT/ncu_raw" "$OUT/figures"

if [[ ! -x "$BIN" ]]; then
    echo "[profile_matrix] missing executable: $BIN" >&2
    exit 2
fi

COMMON=(--batch "$BATCH" --seq "$SEQ" --heads "$HEADS" --normalize true --dist normal --check false)

# Device attributes are obtained from the same CUDA runtime and GPU as the run.
"$BIN" "${COMMON[@]}" --head_dim 64 --dtype fp16 --warmup 1 --iters 1 \
    --device_json "$OUT/gpu_info.json"

{
    date --iso-8601=seconds
    echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
    echo "host=$(hostname)"
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,clocks.max.sm,clocks.max.memory --format=csv,noheader
    nvcc --version
    command -v ncu >/dev/null 2>&1 && ncu --version || true
    command -v nsys >/dev/null 2>&1 && nsys --version || true
} > "$OUT/environment.txt" 2>&1

# CUDA-event timings without profiler overhead.
TIMING_CSV="$OUT/timings.csv"
rm -f "$TIMING_CSV"
for dtype in fp16 bf16; do
    for head_dim in 64 128 256; do
        "$BIN" "${COMMON[@]}" --dtype "$dtype" --head_dim "$head_dim" \
            --warmup 20 --iters 100 --csv "$TIMING_CSV"
    done
done

# A representative Systems trace exposes launches, API overhead, and gaps.
if command -v nsys >/dev/null 2>&1; then
    nsys stats --help-reports > "$OUT/nsys_available_reports.txt" 2>&1 || true
    NSYS_STEM="$OUT/nsys_fp16_hd128_tokens${TOKENS}"
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$NSYS_STEM" \
        "$BIN" "${COMMON[@]}" --dtype fp16 --head_dim 128 --warmup 5 --iters 50 || true
    NSYS_REP="$NSYS_STEM.nsys-rep"
    if [[ -f "$NSYS_REP" ]]; then
        if ! nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
            --format csv --output "$OUT/nsys_stats" "$NSYS_REP"; then
            nsys stats --force-export=true --report gpukernsum,cudaapisum \
                --format csv --output "$OUT/nsys_stats" "$NSYS_REP" || true
        fi
        if [[ -f "$NSYS_STEM.sqlite" ]]; then
            python3 "$ROOT/scripts/summarize_nsys.py" --sqlite "$NSYS_STEM.sqlite" \
                --output-dir "$OUT"
            if python3 -c 'import matplotlib' >/dev/null 2>&1; then
                python3 "$ROOT/scripts/plot_nsys.py" --sqlite "$NSYS_STEM.sqlite" \
                    --output "$OUT/figures/nsys_timeline.png"
            fi
        fi
    fi
else
    echo "[profile_matrix] nsys unavailable" | tee "$OUT/nsys_unavailable.txt"
fi

# Full sets include Occupancy, MemoryWorkloadAnalysis, WarpStateStats and the
# native metrics required by scripts/parse_ncu.py. Only the timed launch is
# profiled: launch 0 is the benchmark's dispatch-validation launch.
if command -v ncu >/dev/null 2>&1; then
    for dtype in fp16 bf16; do
        for head_dim in 64 128 256; do
            stem="${dtype}_hd${head_dim}_tokens${TOKENS}"
            report="$OUT/ncu_$stem"
            log="$OUT/ncu_$stem.log"
            echo "[profile_matrix] ncu $stem"
            if ncu --set full --replay-mode kernel --clock-control none \
                --launch-skip 1 --launch-count 1 \
                --force-overwrite -o "$report" \
                "$BIN" "${COMMON[@]}" --dtype "$dtype" --head_dim "$head_dim" \
                --warmup 0 --iters 1 >"$log" 2>&1; then
                if ! ncu --import "$report.ncu-rep" --csv --page raw \
                    > "$OUT/ncu_raw/$stem.csv" 2>>"$log"; then
                    echo "[profile_matrix] failed to export $stem" | tee -a "$log" >&2
                    NCU_FAILURES=$((NCU_FAILURES + 1))
                fi
            else
                echo "[profile_matrix] failed to profile $stem (see $log)" >&2
                NCU_FAILURES=$((NCU_FAILURES + 1))
            fi
        done
    done
else
    echo "[profile_matrix] ncu unavailable" | tee "$OUT/ncu_unavailable.txt"
    NCU_FAILURES=6
fi

if compgen -G "$OUT/ncu_raw/*.csv" >/dev/null; then
    python3 "$ROOT/scripts/parse_ncu.py" --input-dir "$OUT/ncu_raw" \
        --device-json "$OUT/gpu_info.json" --output "$OUT/ncu_metrics.csv"
else
    python3 "$ROOT/scripts/timings_to_metrics.py" --timings "$TIMING_CSV" \
        --device-json "$OUT/gpu_info.json" --output "$OUT/profile_metrics.csv"
fi

METRICS="$OUT/ncu_metrics.csv"
[[ -f "$METRICS" ]] || METRICS="$OUT/profile_metrics.csv"
if python3 -c 'import matplotlib' >/dev/null 2>&1; then
    python3 "$ROOT/scripts/plot_profiles.py" --metrics "$METRICS" \
        --output-dir "$OUT/figures"
else
    echo "[profile_matrix] matplotlib unavailable; plot later on login node" >&2
fi

echo "ncu_failures=$NCU_FAILURES" > "$OUT/profile_status.txt"
if [[ "$NCU_FAILURES" -gt 0 ]]; then
    echo "[profile_matrix] warning: $NCU_FAILURES NCU configurations failed; timing/NSYS/fallback outputs were retained" >&2
fi
echo "[profile_matrix] output=$OUT ncu_failures=$NCU_FAILURES"
