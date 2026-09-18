#!/usr/bin/env bash
#SBATCH --job-name=hadamard-report
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gz2522/gz2522/tmp/hadamard/runs/slurm/%x_%j.err

set -euo pipefail
ROOT="${HADAMARD_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TC="$ROOT/tensor_core"
: "${TIMING_JOB:?set TIMING_JOB}"
: "${VERIFY_JOB:?set VERIFY_JOB}"
PYTHON="${HADAMARD_TEST_PYTHON:-/scratch/gz2522/gz2522/tmp/micro-hackthon/env/dynax-py312/bin/python}"
TIMING="$TC/results/audit_$TIMING_JOB"
if [[ "${TIMING_KIND:-audit}" == verify ]]; then
    TIMING="$TC/results/verify_$TIMING_JOB"
fi
VERIFY="$TC/results/verify_$VERIFY_JOB"
PROFILE="$VERIFY"
if [[ -n "${PROFILE_JOB:-}" ]]; then
    PROFILE="$TC/results/ncu_$PROFILE_JOB"
fi
OUT="$TIMING/final"
mkdir -p "$OUT"
# Nominal SXM roofs only when saved device/power-cap identifies the actual GPU.
PEAK_ARGS=()
if "$PYTHON" -c 'from pathlib import Path; import sys; s=Path(sys.argv[1]).read_text(); sys.exit(not ("NVIDIA H200" in s and "/  700W" in s))' "$PROFILE/environment.txt"; then
    PEAK_ARGS=(--dram-peak-gbs 4800 --fp32-peak-tflops 67)
elif "$PYTHON" -c 'from pathlib import Path; import sys; s=Path(sys.argv[1]).read_text(); sys.exit(not ("NVIDIA H100 80GB HBM3" in s and "/  700W" in s))' "$PROFILE/environment.txt"; then
    PEAK_ARGS=(--dram-peak-gbs 3350 --fp32-peak-tflops 67)
fi
"$PYTHON" "$TC/scripts/summarize_audit.py" --timing-dir "$TIMING" --verify-dir "$VERIFY" \
    --profile-dir "$PROFILE" --output-dir "$OUT" "${PEAK_ARGS[@]}" > "$OUT/generation.log" 2>&1
echo "Generated $OUT/summary.md; inspect validation failures and NCU availability, not just process exit status."
