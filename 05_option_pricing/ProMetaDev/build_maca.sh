#!/usr/bin/env bash
# ============================================================
# build_maca.sh —— 沐曦（MetaX / 曦云 C500）平台构建
#   用法: bash build_maca.sh
#   产物: ./build_maca/cuda_pricing
#
#   与 build.sh（nvcc + CMake）的区别：
#     1. 沐曦用 mxcc 编译，设备代码带 -x maca、架构用 -offload-arch native；
#     2. CUDA 兼容头文件在 $MACA/tools/cu-bridge/include（把 cudaXxx 声明为
#        wcudaXxx，实现在 $MACA/lib/libruntime_cu.so），必须按 MACA 的
#        cu-bridge/bin/conf.json 里 [link][adder] 追加链接参数；
#     3. -imacros __macro_mxcc.h 让 __CUDACC__ 等 CUDA 宏在 mxcc 下生效；
#     4. 本工程用 cuRAND 生成随机数，沐曦的对应库是 mcRAND：
#        cu-bridge 的 curand_kernel.h 会转去 include <mcrand_kernel.h>，
#        因此必须额外加 -I$MACA/include/mcrand（该路径也正是 MACA 自己的
#        cu-bridge/bin/conf.json 里为 -lcurand 配置的追加项），链接用 -lmcrand。
#     5. fast-math 开关是 -use-fast-math（连字符），与本工程 CMake 里的
#        -use_fast_math 等价。
#
#   本工程用到设备侧 double（__constant__ 参数与 FP64 定价公式），
#   沐曦 C500 实测设备侧 FP64 可用，无需改写。
#
#   测试脚本 tests/run_tests.sh 固定用 ./build/cuda_pricing，
#   可先 `mkdir -p build && cp build_maca/cuda_pricing build/` 再跑。
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
echo "[build_maca] 编译 cuda_pricing（4 个源文件，一次编译链接）"
"$MXCC" -x maca -offload-arch native --maca-path="$MACA" \
    -I"$ROOT/include" -I"$BRIDGE_INC" -I"$MACA/include/mcrand" -L"$MACA/lib" \
    -imacros __macro_mxcc.h -forward-unknown-to-compiler \
    -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    -lmcrand \
    -Wno-unused-command-line-argument \
    -std=c++17 -O3 -use-fast-math \
    src/main.cpp src/bs_formula.cpp src/file_io.cpp src/pricing_kernels.cu \
    -o "$BUILD_DIR/cuda_pricing"

echo "构建完成: $ROOT/$BUILD_DIR/cuda_pricing"
echo "测试: mkdir -p build && cp $BUILD_DIR/cuda_pricing build/ && bash tests/run_tests.sh"
