#!/usr/bin/env bash
# A/B sweep: baseline, shuffle/vectorized FHT, unfused INT4, fused INT4.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HADAMARD_ADVANCED_BIN:-$ROOT/build/hadamard_advanced_bench}"
OUT="${HADAMARD_ADVANCED_OUT:-$ROOT/results/advanced_run}"
CSV="$OUT/advanced_results.csv"
mkdir -p "$OUT/figures"
rm -f "$CSV"

if [[ ! -x "$BIN" ]]; then
    make -C "$ROOT"
fi

for dtype in fp16 bf16; do
    for head_dim in 64 128 256; do
        "$BIN" --batch 4 --seq 1024 --heads 32 --head_dim "$head_dim" \
            --dtype "$dtype" --normalize true --warmup 20 --iters 100 \
            --csv "$CSV"
    done
done

if python3 -c 'import matplotlib' >/dev/null 2>&1; then
    python3 "$ROOT/scripts/plot_advanced.py" --input "$CSV" \
        --output "$OUT/figures/advanced_performance.png"
else
    echo "[run_advanced] matplotlib unavailable; plot the retained CSV later" >&2
fi
python3 "$ROOT/scripts/summarize_advanced.py" --input "$CSV" \
    --output "$OUT/README.md" --environment "$OUT/environment.txt"
echo "[run_advanced] results=$CSV"
