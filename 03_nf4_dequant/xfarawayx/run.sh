#!/bin/bash
# ============================================================
# NF4 反量化 —— 统一流程脚本
#
# 用法:
#   ./run.sh [子命令] [选项]
#
# 子命令:
#   generate   仅生成测试数据
#   build      仅编译 CUDA kernel
#   test       编译 → 运行 kernel → 验证正确性 (默认)
#   bench      bitsandbytes 基准性能测试
#   all        数据生成 → 编译 → 运行 → 验证 → bnb 对比
#
# 选项:
#   --rows R            矩阵行数 (默认: 4096)
#   --cols C            矩阵列数 (默认: 4096)
#   --blocksize B       量化块大小 (默认: 64)
#   --compute_type T    bf16|fp16 (默认: bf16)
#   --seed S            随机种子 (默认: 42)
#   --gpu_arch A        GPU 架构, 如 80/89/90 (默认: 自动检测)
#   --warmup W          预热次数 (默认: 10)
#   --repeats N         计时重复次数 (默认: 100)
#   --sweep             bench 时扫描多种矩阵大小
# ============================================================
set -e
set -o pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="${PROJ_DIR}/kernel"
SCRIPTS_DIR="${PROJ_DIR}/scripts"
BUILD_DIR="${KERNEL_DIR}/build"
DATA_DIR="${PROJ_DIR}/data"

# ---------- 自动查找 Python ----------
# 优先使用环境变量 PYTHON，其次使用 venv (含所需依赖)，再回退到系统 python3
if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    : # 使用用户提供的 PYTHON
elif [ -x "${PROJ_DIR}/.venv/bin/python" ]; then
    PYTHON="${PROJ_DIR}/.venv/bin/python"
elif [ -n "${HOME:-}" ] && [ -x "${HOME}/.venv/bin/python" ]; then
    PYTHON="${HOME}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] 找不到 Python, 请设置 PYTHON 环境变量"
    exit 1
fi

# ---------- 默认参数 ----------
ROWS=4096
COLS=4096
BLOCKSIZE=64
COMPUTE_TYPE="bf16"
SEED=42
GPU_ARCH=""
WARMUP=10
REPEATS=100
SWEEP=""

# ---------- 解析子命令 ----------
COMMAND="test"
if [[ "$#" -gt 0 && ! "$1" == --* ]]; then
    COMMAND="$1"
    shift
fi

# ---------- 解析选项 ----------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rows)         ROWS="$2"; shift ;;
        --cols)         COLS="$2"; shift ;;
        --blocksize)    BLOCKSIZE="$2"; shift ;;
        --compute_type) COMPUTE_TYPE="$2"; shift ;;
        --seed)         SEED="$2"; shift ;;
        --gpu_arch)     GPU_ARCH="$2"; shift ;;
        --warmup)       WARMUP="$2"; shift ;;
        --repeats)      REPEATS="$2"; shift ;;
        --sweep)        SWEEP="--sweep" ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

TAG="${ROWS}x${COLS}_bs${BLOCKSIZE}"
WEIGHT_FILE="${DATA_DIR}/nf4_weights_${TAG}.bin"
REF_FILE="${DATA_DIR}/nf4_ref_output_${TAG}_${COMPUTE_TYPE}.bin"
CUDA_OUTPUT="${DATA_DIR}/cuda_output_${TAG}_${COMPUTE_TYPE}.bin"

# ============================================================
# 阶段函数
# ============================================================

do_generate() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Step 1] 生成测试数据"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f "${WEIGHT_FILE}" ] && [ -f "${REF_FILE}" ]; then
        echo "  数据已存在: ${TAG}, 跳过 (删除 data/ 可强制重新生成)"
    else
        ${PYTHON} "${SCRIPTS_DIR}/generate_data.py" \
            --rows ${ROWS} --cols ${COLS} --blocksize ${BLOCKSIZE} \
            --seed ${SEED} --compute_type ${COMPUTE_TYPE} --outdir "${DATA_DIR}"
    fi
}

do_build() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Step 2] 编译 CUDA kernel"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "${BUILD_DIR}"

    local cmake_args="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
    if [ -n "${GPU_ARCH}" ]; then
        cmake_args="${cmake_args} -DGPU_ARCH=${GPU_ARCH}"
    fi

    cd "${BUILD_DIR}"
    cmake .. ${cmake_args} 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -5
    cd "${PROJ_DIR}"

    echo "  可执行文件: ${BUILD_DIR}/nf4_dequant"
}

do_test() {
    do_generate
    do_build

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Step 3] 运行 CUDA kernel"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "${BUILD_DIR}/nf4_dequant" \
        "${WEIGHT_FILE}" "${CUDA_OUTPUT}" "${COMPUTE_TYPE}" ${WARMUP} ${REPEATS}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Step 4] 验证正确性"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ${PYTHON} "${SCRIPTS_DIR}/verify.py" \
        --weight_file "${WEIGHT_FILE}" \
        --ref_file "${REF_FILE}" \
        --cuda_file "${CUDA_OUTPUT}" \
        --compute_type ${COMPUTE_TYPE}
}

do_bench() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  bitsandbytes 基准性能测试"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ${PYTHON} "${SCRIPTS_DIR}/bench_bnb.py" \
        --rows ${ROWS} --cols ${COLS} --blocksize ${BLOCKSIZE} \
        --seed ${SEED} --warmup ${WARMUP} --repeats ${REPEATS} ${SWEEP}
}

do_all() {
    do_test
    do_bench
}

# ============================================================
# 入口
# ============================================================

echo "============================================"
echo "  NF4 反量化测试"
echo "  矩阵: ${ROWS} x ${COLS}, 块大小: ${BLOCKSIZE}"
echo "  输出类型: ${COMPUTE_TYPE}, 命令: ${COMMAND}"
echo "============================================"

case ${COMMAND} in
    generate) do_generate ;;
    build)    do_build ;;
    test)     do_test ;;
    bench)    do_bench ;;
    all)      do_all ;;
    *)
        echo "[ERROR] 未知子命令: ${COMMAND}"
        echo "可用子命令: generate | build | test | bench | all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  完成: ${COMMAND}"
echo "============================================"
