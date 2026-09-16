#!/usr/bin/env bash
# One entry for all platforms; outputs are excluded from the submission tree.
set -euo pipefail
cd "$(dirname "$0")/.."

export PLATFORM="${PLATFORM:-nvidia}"
export RESULT_DIR="${RESULT_DIR:-output/results/${PLATFORM}}"
export WORK_DIR="${WORK_DIR:-output/work/${PLATFORM}}"
export ARCH="${ARCH:-}"
SIZES="${SIZES:-1920x1080,3840x2160}"
CHANNELS="${CHANNELS:-1,3}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
CPU_REPEAT="${CPU_REPEAT:-3}"
VALIDATE_SIZE="${VALIDATE_SIZE:-1920x1080}"
export SIZE="${QUALITY_SIZE:-1920x1080}"

case "${PLATFORM}" in
    nvidia)
        if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1 && test -x /usr/local/cuda/bin/nvcc; then
            export PATH="/usr/local/cuda/bin:${PATH}"
        fi ;;
    iluvatar)
        export COREX_HOME="${COREX_HOME:-/usr/local/corex}"
        export PATH="${COREX_HOME}/bin:${PATH}" ;;
    metax)
        export MACA_HOME="${MACA_HOME:-/opt/maca}"
        export PATH="${MACA_HOME}/bin:${PATH}" ;;
    moore) export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}" ;;
    *) echo "Unsupported PLATFORM: ${PLATFORM}" >&2; exit 2 ;;
esac
mkdir -p "${RESULT_DIR}" "${WORK_DIR}"
BIN="./build/${PLATFORM}/nlm_denoise"

echo '[1/7] Environment and build'
bash scripts/collect_environment.sh "${RESULT_DIR}/environment.txt"
make build -j"$(nproc)" 2>&1 | tee "${RESULT_DIR}/build.log"

echo '[2/7] CPU self checks and three-version GPU validation'
"${BIN}" units 2>&1 | tee "${RESULT_DIR}/self_check.log"
"${BIN}" gen -o "${WORK_DIR}/validation.png" --size "${VALIDATE_SIZE}" --channels 3 --sigma 25 --seed 7
"${BIN}" validate -i "${WORK_DIR}/validation.png" -p params.txt \
    -o "${WORK_DIR}/validated.png" 2>&1 | tee "${RESULT_DIR}/validation.log"

echo '[3/7] GPU matrix'
"${BIN}" bench --sizes "${SIZES}" --channels "${CHANNELS}" --param-sets all \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --log "${RESULT_DIR}/benchmark.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark.log"

echo '[4/7] CPU baseline on the same input and parameters'
"${BIN}" bench --sizes "${VALIDATE_SIZE}" --channels 3 --param-sets base \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --with-cpu --cpu-repeat "${CPU_REPEAT}" \
    --log "${RESULT_DIR}/cpu_baseline.csv" 2>&1 | tee "${RESULT_DIR}/cpu_baseline.log"

echo '[5/7] Paired-image quality sweeps (V2 on every platform)'
CHANNELS=3 bash scripts/run_quality_sweep.sh 2>&1 | tee "${RESULT_DIR}/quality.log"

echo '[6/7] Optional device profiler'
if [[ "${PLATFORM}" == "nvidia" ]] && command -v nsys >/dev/null 2>&1; then
    nsys profile -f true -o "${WORK_DIR}/nsys_v2" "${BIN}" run \
        -i "${WORK_DIR}/validation.png" -o "${WORK_DIR}/profiled.png" -p params.txt --kernel 2 --log /dev/null
    nsys stats "${WORK_DIR}/nsys_v2.nsys-rep" --report cuda_gpu_kern_sum \
        --report cuda_gpu_mem_time_sum > "${RESULT_DIR}/profiler.txt"
elif [[ "${PLATFORM}" == "metax" ]] && test -x "${MACA_HOME}/bin/mcTracer"; then
    mkdir -p "${RESULT_DIR}/mctracer"
    "${MACA_HOME}/bin/mcTracer" --mctx --odname "${RESULT_DIR}/mctracer" --name nlm_v1 \
        "${BIN}" run -i "${WORK_DIR}/validation.png" -o "${WORK_DIR}/profiled.png" \
        -p params.txt --kernel 1 --log /dev/null 2>&1 | tee "${RESULT_DIR}/profiler.txt"
else
    echo 'SKIPPED: no supported profiler CLI found; this run provides no new hardware profiling evidence.' \
        | tee "${RESULT_DIR}/profiler.txt"
fi

echo '[7/7] Hash evidence'
(cd "${RESULT_DIR}"; find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
echo "Core validation and benchmark complete: ${RESULT_DIR}; profiler status: ${RESULT_DIR}/profiler.txt"
