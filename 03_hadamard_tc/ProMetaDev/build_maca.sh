#!/usr/bin/env bash
# ============================================================
# build_maca.sh —— 沐曦（MetaX / 曦云 C500）平台构建
#   用法: bash build_maca.sh
#   产物: ./build_maca/hadamard
#
#   与 build.sh（nvcc + CMake）的区别：
#     1. 沐曦用 mxcc 编译，设备代码带 -x maca、架构用 -offload-arch native；
#     2. CUDA 兼容头文件在 $MACA/tools/cu-bridge/include（把 cudaXxx 声明为
#        wcudaXxx，实现在 $MACA/lib/libruntime_cu.so），因此必须按 MACA 的
#        cu-bridge/bin/conf.json 里 [link][adder] 追加链接参数，否则链接期报
#        undefined reference to `wcudaMalloc' 之类；
#     3. 测试脚本支持 BIN= 覆盖，可直接 `BIN=./build_maca/hadamard bash tests/run_tests.sh`。
#   运行前需 export LD_LIBRARY_PATH=$MACA/lib
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MACA="${MACA_PATH:-}"
if [ -z "$MACA" ]; then
    for d in /opt/maca /opt/maca-* /usr/local/maca /usr/local/maca-*; do
        [ -x "$d/mxgpu_llvm/bin/mxcc" ] && { MACA="$d"; break; }
    done
fi
[ -n "$MACA" ] || { echo "ERROR: 未找到 MACA SDK（需要 \$MACA/mxgpu_llvm/bin/mxcc）" >&2; exit 2; }

MXCC="$MACA/mxgpu_llvm/bin/mxcc"
BRIDGE_INC="$MACA/tools/cu-bridge/include"
BUILD_DIR="${BUILD_DIR:-build_maca}"
[ -d "$BRIDGE_INC" ] || { echo "ERROR: 缺少 $BRIDGE_INC" >&2; exit 2; }

export LD_LIBRARY_PATH="$MACA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "---------------------------------------------"
echo "[build_maca] MACA SDK = $MACA"
echo "[build_maca] mxcc     = $("$MXCC" --version 2>&1 | head -1)"
command -v mx-smi >/dev/null 2>&1 && echo "[build_maca] 设备     = $(mx-smi 2>/dev/null | grep -oE 'MetaX [A-Za-z0-9]+' | head -1)"
echo "---------------------------------------------"

mkdir -p "$BUILD_DIR"
echo "[build_maca] 编译 hadamard（4 个源文件，一次编译链接）"
"$MXCC" -x maca -offload-arch native --maca-path="$MACA" \
    -I"$ROOT/include" -I"$BRIDGE_INC" -L"$MACA/lib" \
    -imacros __macro_mxcc.h -forward-unknown-to-compiler \
    -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    -Wno-unused-command-line-argument \
    -std=c++17 -O3 -use-fast-math \
    src/main.cpp src/io.cpp src/cpu_ref.cpp src/fht.cu \
    -o "$BUILD_DIR/hadamard"

echo "构建完成: $ROOT/$BUILD_DIR/hadamard"
echo "测试: BIN=./$BUILD_DIR/hadamard bash tests/run_tests.sh"
