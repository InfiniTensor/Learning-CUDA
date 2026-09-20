#!/usr/bin/env bash
# ============================================================
# build.sh —— 一键构建 CudaNBodyGravitySim_2026summer
#   用法:  bash build.sh [--clean]
#   产物:  ./build/nbody
#   特性:
#     - 自动探测 CUDA: $CUDA_HOME env > which nvcc > /usr/local/cuda*
#     - 探测失败时给出详细诊断 (which nvcc / ldconfig / ls /usr/local)
#     - 自动探测 GPU SM 并打印 nvcc/g++ 版本
#     - 不再依赖任何 patch 脚本 (main.cpp / nbody_kernels.cu 头文件独立)
# ============================================================
set -e
cd "$(dirname "$0")"
ROOT="$PWD"

# ---- 参数: --clean 删除 build 目录 ----
if [ "${1:-}" = "--clean" ]; then
    echo "[build] --clean: removing build/"
    rm -rf build
fi

# ============================================================
# 1. CUDA Toolkit 路径自动探测 (优先级: env > which > 常见目录)
# ============================================================
CUDA_HOME=""
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[build] using \$CUDA_HOME env = $CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    # 从 nvcc 路径反推 CUDA_HOME: $(dirname $(dirname $(which nvcc)))
    NVCC_PATH="$(command -v nvcc)"
    CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
    echo "[build] auto-detect CUDA_HOME from \$(which nvcc) -> $CUDA_HOME"
else
    for p in /usr/local/cuda-13 /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12.3 \
             /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-12 \
             /usr/local/cuda-11.8 /usr/local/cuda-11 /usr/local/cuda \
             /opt/cuda /usr/cuda; do
        if [ -x "$p/bin/nvcc" ]; then
            CUDA_HOME="$p"; break
        fi
    done
fi

if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo ""
    echo "[ERROR] ======================================================"
    echo "[ERROR] 找不到 nvcc (CUDA 编译器). 请安装 CUDA Toolkit."
    echo "[ERROR]   \$CUDA_HOME=${CUDA_HOME:-(未设置)}"
    echo "[ERROR]   which nvcc: $(command -v nvcc 2>&1 || echo '(不存在)')"
    echo "[ERROR]   ls /usr/local/cuda*:"
    ls -d /usr/local/cuda* 2>/dev/null || echo "    (无 /usr/local/cuda*)"
    echo "[ERROR]   ldconfig -p | grep cuda (前 8 行):"
    ldconfig -p 2>/dev/null | grep -i cuda | head -8 || echo "    (无 ldconfig 结果)"
    echo "[ERROR] ======================================================"
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ============================================================
# 2. 打印版本信息 (nvcc, g++, detect_sm)
# ============================================================
echo "---------------------------------------------"
echo "[build] nvcc:   $("$CUDA_HOME/bin/nvcc" --version | grep -E "release|V[0-9]" | head -1)"
echo "[build] g++ :   $(g++ --version | head -1)"
echo "[build] cmake:  $(cmake --version | head -1)"
SM_LIST="$(bash "$ROOT/scripts/detect_sm.sh")"
echo "[build] target SM list = $SM_LIST"
export NBODY_SM_LIST="$SM_LIST"
echo "---------------------------------------------"

# ============================================================
# 3. CMake Configure + Build
# ============================================================
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DNBODY_SM_LIST="$SM_LIST" \
      .. 2>&1 | tail -15

cmake --build . -j"$(nproc)" 2>&1 | tail -20
cd "$ROOT"

# ============================================================
# 4. 收尾验证
# ============================================================
echo "---- build done ----"
if [ -x ./build/nbody ]; then
    ls -la ./build/nbody
    echo "OK: ./build/nbody"
else
    echo "[ERROR] ./build/nbody 不存在! 请查看上面的 cmake / make 错误日志"
    exit 2
fi
