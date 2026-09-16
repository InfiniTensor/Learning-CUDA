#!/usr/bin/env bash
# Reproduce quality experiments on paired deterministic synthetic images.
# The clean/noisy pair always uses the same seed; only noise amplitude changes.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT_DIR="${RESULT_DIR:-output/results}"
WORK_DIR="${WORK_DIR:-output/work}"
PLATFORM="${PLATFORM:-nvidia}"
ARCH="${ARCH:-}"
SEED="${SEED:-7}"
SIZE="${SIZE:-1920x1080}"
CHANNELS="${CHANNELS:-3}"

mkdir -p "${RESULT_DIR}" "${WORK_DIR}"
rm -f "${RESULT_DIR}/quality_tradeoff.csv" "${RESULT_DIR}/quality_sigma.csv"

if [[ "${PLATFORM}" == "nvidia" ]]; then
    make build PLATFORM="${PLATFORM}" ARCH="${ARCH}" -j"$(nproc)"
else
    make build PLATFORM="${PLATFORM}" -j"$(nproc)"
fi
BIN="./build/${PLATFORM}/nlm_denoise"

CLEAN="${WORK_DIR}/clean_${SIZE}_${CHANNELS}ch.png"
"${BIN}" gen -o "${CLEAN}" --size "${SIZE}" \
    --channels "${CHANNELS}" --sigma 0 --seed "${SEED}"

# A fair small/base/large comparison: same sigma=25 input, h=10, only pr/sr change.
NOISY25="${WORK_DIR}/noisy_${SIZE}_${CHANNELS}ch_sigma25.png"
"${BIN}" gen -o "${NOISY25}" --size "${SIZE}" \
    --channels "${CHANNELS}" --sigma 25 --seed "${SEED}"
"${BIN}" metrics --reference "${CLEAN}" --test "${NOISY25}" \
    --label noisy-sigma25 --log "${RESULT_DIR}/quality_tradeoff.csv"

for NAME in small base large; do
    OUTPUT="${WORK_DIR}/denoised_${NAME}_${SIZE}_${CHANNELS}ch_sigma25.png"
    PARAM_FILE="scripts/params/${NAME}.txt"
    if [[ "${NAME}" == "base" ]]; then PARAM_FILE=params.txt; fi
    "${BIN}" run -i "${NOISY25}" -o "${OUTPUT}" \
        -p "${PARAM_FILE}" --kernel 2 --log /dev/null
    "${BIN}" metrics --reference "${CLEAN}" --test "${OUTPUT}" \
        --label "${NAME}" --log "${RESULT_DIR}/quality_tradeoff.csv"
done

# Noise-strength sweep: parameter sigma matches the generated input amplitude.
for SIGMA in 10 25 50; do
    NOISY="${WORK_DIR}/noisy_${SIZE}_${CHANNELS}ch_sigma${SIGMA}.png"
    OUTPUT="${WORK_DIR}/denoised_base_${SIZE}_${CHANNELS}ch_sigma${SIGMA}.png"
    "${BIN}" gen -o "${NOISY}" --size "${SIZE}" \
        --channels "${CHANNELS}" --sigma "${SIGMA}" --seed "${SEED}"
    "${BIN}" metrics --reference "${CLEAN}" --test "${NOISY}" \
        --label "noisy-sigma${SIGMA}" --log "${RESULT_DIR}/quality_sigma.csv"
    PARAM_FILE="scripts/params/sigma${SIGMA}.txt"
    if [[ "${SIGMA}" == "25" ]]; then PARAM_FILE=params.txt; fi
    "${BIN}" run -i "${NOISY}" -o "${OUTPUT}" \
        -p "${PARAM_FILE}" --kernel 2 --log /dev/null
    "${BIN}" metrics --reference "${CLEAN}" --test "${OUTPUT}" \
        --label "denoised-sigma${SIGMA}" --log "${RESULT_DIR}/quality_sigma.csv"
done

echo "Quality results: ${RESULT_DIR}/quality_tradeoff.csv, ${RESULT_DIR}/quality_sigma.csv"
echo "Generated images: ${WORK_DIR}/"
