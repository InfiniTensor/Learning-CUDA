#!/usr/bin/env bash
# ============================================================================
# 沐曦（MetaX / 曦云 C500）平台构建脚本
#   用法: bash build_maca.sh
#   依赖: MACA SDK（默认 /opt/maca，可用 MACA_PATH 覆盖）
#
#   与 build.sh（NVIDIA/nvcc）的区别：
#     1. 沐曦用 mxcc 编译，且设备代码必须带 -x maca、架构用 -offload-arch native；
#     2. CUDA 兼容头文件在 $MACA/tools/cu-bridge/include（cuda_runtime.h、cublas_v2.h 等），
#        它把 cuda* API 映射为 wcuda*，实现体在 $MACA/lib/libruntime_cu.so；
#        因此链接必须带上 MACA 的 cu-bridge 配置 conf.json 里 [link][adder] 指定的那组参数，
#        否则会报 undefined reference to `wcudaMalloc' 之类的错误。
#     3. 运行可执行文件前需要 LD_LIBRARY_PATH=$MACA/lib（mx-smi 等工具同理）。
#
#   本平台的 64 位原子操作实测可用（atomicAdd(ull*) 与 CAS 均正确），
#   故本文件不需要 PLATFORM_ILUVATAR/PLATFORM_MACA 之类的分支。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 自动探测 MACA SDK ----------
MACA="${MACA_PATH:-}"
if [ -z "$MACA" ]; then
    for d in /opt/maca /opt/maca-* /usr/local/maca /usr/local/maca-*; do
        if [ -x "$d/mxgpu_llvm/bin/mxcc" ]; then
            MACA="$d"
            break
        fi
    done
fi
if [ -z "$MACA" ]; then
    echo "ERROR: 未找到 MACA SDK（需要 \$MACA/mxgpu_llvm/bin/mxcc）" >&2
    echo "       可通过 MACA_PATH=/path/to/maca 指定" >&2
    exit 2
fi

MXCC="$MACA/mxgpu_llvm/bin/mxcc"
ROOT="$(pwd)"
BRIDGE_INC="$MACA/tools/cu-bridge/include"
BUILD_DIR="${BUILD_DIR:-build_maca}"
SRCS="src/main.cpp src/io.cpp src/cpu_ref.cpp src/gpu_maxflow.cu"

if [ ! -d "$BRIDGE_INC" ]; then
    echo "ERROR: 未找到 CUDA 兼容头文件目录 $BRIDGE_INC" >&2
    exit 2
fi

export LD_LIBRARY_PATH="$MACA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[build_maca] MACA SDK    = $MACA"
echo "[build_maca] 编译器       = $("$MXCC" --version 2>&1 | head -1)"
if command -v mx-smi >/dev/null 2>&1; then
    echo "[build_maca] 设备          = $(mx-smi 2>/dev/null | grep -m1 -oE 'MetaX [A-Za-z0-9]+' || echo '未知')"
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "[build_maca] 编译（-x maca，含 4 个源文件一次性编译链接）"
"$MXCC" -x maca -offload-arch native --maca-path="$MACA" \
    -I"$ROOT/include" -I"$BRIDGE_INC" -L"$MACA/lib" \
    -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    -Wno-unused-command-line-argument \
    -std=c++17 -O3 \
    $SRCS -o "$BUILD_DIR/maxflow"

echo "构建完成: $(pwd)/$BUILD_DIR/maxflow"
echo "运行前请设置: export LD_LIBRARY_PATH=$MACA/lib:\$LD_LIBRARY_PATH"
