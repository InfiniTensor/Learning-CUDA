#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="representative"
if [[ "${1:-}" == "--full" ]]; then
    MATRIX="full"
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--full]" >&2
    exit 2
fi

if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]] || ! "${PROJECT_DIR}/.venv/bin/python" -c "import torch; from fast_hadamard_transform import hadamard_transform" >/dev/null 2>&1; then
    "${PROJECT_DIR}/scripts/bootstrap_env.sh"
fi

PYTHON="${PROJECT_DIR}/.venv/bin/python"
RESULT_TAG="$(date -u +%Y%m%dT%H%M%SZ)_${MATRIX}"
RESULT_DIR="${PROJECT_DIR}/results/${RESULT_TAG}"
DAO_DIR="${PROJECT_DIR}/third_party/fast-hadamard-transform"
DAO_COMMIT="unknown"
if [[ -d "${DAO_DIR}/.git" ]]; then
    DAO_COMMIT="$(git -C "${DAO_DIR}" rev-parse HEAD)"
fi

mkdir -p "${RESULT_DIR}"
cd "${PROJECT_DIR}"

echo "[suite] Building versioned CUDA kernels"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel
ctest --test-dir build --output-on-failure

echo "[suite] Correctness gate: v0/v1/v2, FP32/FP16/BF16, D=128/1024/8192"
"${PYTHON}" benchmark/correctness.py --batch 4 --dim 128 1024 8192 --dtype all --kernel all | tee "${RESULT_DIR}/correctness.log"

echo "[suite] Native CUDA Event trials (${MATRIX})"
"${PYTHON}" benchmark/benchmark_native.py --matrix "${MATRIX}" --warmup 100 --repeat 1000 --trials 3 --binary "${PROJECT_DIR}/build/benchmark_cuda" --raw-csv "${RESULT_DIR}/native_raw.csv" --summary-csv "${RESULT_DIR}/native_summary.csv"

echo "[suite] Same-process v0/v1/v2/Dao/clone trials (${MATRIX})"
"${PYTHON}" benchmark/benchmark_fht.py --matrix "${MATRIX}" --dtype all --implementation all --warmup 100 --repeat 1000 --trials 3 --library "${PROJECT_DIR}/build/libhadamard.so" --raw-csv "${RESULT_DIR}/same_process_raw.csv" --summary-csv "${RESULT_DIR}/same_process_summary.csv" --metadata "${RESULT_DIR}/metadata.json" --dao-commit "${DAO_COMMIT}"

"${PYTHON}" benchmark/report_results.py --metadata "${RESULT_DIR}/metadata.json" --same-summary "${RESULT_DIR}/same_process_summary.csv" --native-summary "${RESULT_DIR}/native_summary.csv" --summary "${RESULT_DIR}/summary.csv" --report "${RESULT_DIR}/report.md"

ln -sfn "${RESULT_TAG}" "${PROJECT_DIR}/results/latest"
echo "[suite] Complete: ${RESULT_DIR}/report.md"
