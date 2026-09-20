#!/usr/bin/env bash
# Collect Nsight Compute / Nsight Systems reports for the quantize and
# dequantize kernels. Output .ncu-rep / .nsys-rep files are portable: they
# can be opened on a Windows host with the Nsight Compute / Nsight Systems
# GUI (install a version >= the collecting CLI).
#
# Usage: profile.sh /path/to/lowp TENSOR.bin CONFIG.txt OUTDIR
#
# Outputs in OUTDIR:
#   <tensor>_kernels.ncu-rep    per-kernel metrics (Nsight Compute)
#   <tensor>_ncu_summary.txt    full "details" page dump of that report
#   <tensor>_ncu.log            ncu collection log (diagnoses permission errors)
#   <tensor>_pipeline.nsys-rep  whole-pipeline timeline (Nsight Systems)
#   <tensor>_nsys_stats.txt     nsys stats dump
set -euo pipefail

LOWP=${1:?path to lowp binary}
TENSOR=${2:?tensor file}
CONFIG=${3:?config file}
OUT=${4:?output dir}
BIN=$(basename "${TENSOR%.*}")

mkdir -p "$OUT"

# --- Nsight Compute: per-kernel metrics (memory throughput, occupancy...)
# Profile every kernel in the pipeline (amax / quant_* / dequant_* / metrics).
# ncu needs GPU performance-counter permission (ERR_NVGPUCTRPERM on WSL2 or
# hosts without profiling permission); on such hosts this step is skipped
# with a hint instead of failing the whole script. Any other ncu failure is
# fatal so a missing report can never slip through unnoticed.
if command -v ncu >/dev/null 2>&1; then
  NCU_LOG="$OUT/${BIN}_ncu.log"
  if ncu --set full \
        -k 'regex:quant|dequant|amax|metrics' \
        --target-processes all \
        -o "$OUT/${BIN}_kernels" \
        "$LOWP" run --input "$TENSOR" --config "$CONFIG" \
        >"$NCU_LOG" 2>&1; then
    # Dump to a file first: piping straight into `head` would SIGPIPE/kill
    # ncu under `set -o pipefail` once the line limit is reached.
    ncu --import "$OUT/${BIN}_kernels.ncu-rep" --page details \
        > "$OUT/${BIN}_ncu_summary.txt" 2>&1
    head -160 "$OUT/${BIN}_ncu_summary.txt"
  elif grep -q ERR_NVGPUCTRPERM "$NCU_LOG"; then
    echo "ncu skipped: no GPU perf-counter permission (see $NCU_LOG)"
  else
    echo "ncu FAILED (report file may be missing); last 20 log lines:" >&2
    tail -20 "$NCU_LOG" >&2
    exit 1
  fi
else
  echo "ncu skipped: Nsight Compute CLI not found in PATH"
fi

# --- Nsight Systems: timeline of the whole pipeline
if command -v nsys >/dev/null 2>&1; then
  nsys profile -o "$OUT/${BIN}_pipeline" \
      "$LOWP" run --input "$TENSOR" --config "$CONFIG"
  nsys stats "$OUT/${BIN}_pipeline.nsys-rep" \
      > "$OUT/${BIN}_nsys_stats.txt" 2>&1
  head -60 "$OUT/${BIN}_nsys_stats.txt"
else
  echo "nsys skipped: Nsight Systems CLI not found in PATH"
fi

echo "reports written to $OUT (copy to Windows and open with Nsight GUIs)"
