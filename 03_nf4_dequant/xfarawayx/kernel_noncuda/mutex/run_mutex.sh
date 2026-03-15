#!/bin/bash
set -e
set -o pipefail

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL_DIR="${PROJ_DIR}/kernel_noncuda/mutex"
SCRIPTS_DIR="${PROJ_DIR}/scripts"
DATA_DIR="${PROJ_DIR}/data"

if [ -x "${PROJ_DIR}/.venv/bin/python" ]; then
    PYTHON="${PROJ_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] 找不到 python3"
    exit 1
fi

ROWS=4096
COLS=4096
BLOCKSIZE=64
COMPUTE_TYPE="bf16"
WARMUP=10
REPEATS=100
MXCC_BIN="${MXCC:-mxcc}"
COMMAND="test"

if [[ "$#" -gt 0 && ! "$1" == --* ]]; then
    COMMAND="$1"
    shift
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rows) ROWS="$2"; shift ;;
        --cols) COLS="$2"; shift ;;
        --blocksize) BLOCKSIZE="$2"; shift ;;
        --compute_type) COMPUTE_TYPE="$2"; shift ;;
        --warmup) WARMUP="$2"; shift ;;
        --repeats) REPEATS="$2"; shift ;;
        --mxcc) MXCC_BIN="$2"; shift ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

TAG="${ROWS}x${COLS}_bs${BLOCKSIZE}"
WEIGHT_FILE="${DATA_DIR}/nf4_weights_${TAG}.bin"
REF_FILE="${DATA_DIR}/nf4_ref_output_${TAG}_${COMPUTE_TYPE}.bin"
MUTEX_OUTPUT="${DATA_DIR}/mutex_output_${TAG}_${COMPUTE_TYPE}.bin"

build_kernel() {
    echo "[build] 使用编译器: ${MXCC_BIN}"
    make -C "${KERNEL_DIR}" clean >/dev/null
    make -C "${KERNEL_DIR}" MXCC="${MXCC_BIN}" -j"$(nproc)"
}

run_kernel() {
    if [ ! -f "${WEIGHT_FILE}" ]; then
        echo "[ERROR] 缺少权重文件: ${WEIGHT_FILE}"
        echo "        请先在支持 CUDA 的环境执行 ./run.sh generate 生成数据"
        exit 1
    fi

    "${KERNEL_DIR}/nf4_dequant_maca" \
        "${WEIGHT_FILE}" "${MUTEX_OUTPUT}" "${COMPUTE_TYPE}" "${WARMUP}" "${REPEATS}"
}

verify_output() {
    if [ ! -f "${REF_FILE}" ]; then
        echo "[WARN] 缺少参考文件: ${REF_FILE}"
        echo "       跳过 verify。可先在 CUDA 环境运行 ./run.sh generate --compute_type ${COMPUTE_TYPE}"
        return 0
    fi

    "${PYTHON}" "${SCRIPTS_DIR}/verify.py" \
        --weight_file "${WEIGHT_FILE}" \
        --ref_file "${REF_FILE}" \
        --cuda_file "${MUTEX_OUTPUT}" \
        --compute_type "${COMPUTE_TYPE}"
}

case "${COMMAND}" in
    build)
        build_kernel
        ;;
    run)
        run_kernel
        ;;
    verify)
        verify_output
        ;;
    test)
        build_kernel
        run_kernel
        verify_output
        ;;
    *)
        echo "[ERROR] 未知命令: ${COMMAND}"
        echo "可用命令: build | run | verify | test"
        exit 1
        ;;
esac

echo "[DONE] ${COMMAND} 完成"
