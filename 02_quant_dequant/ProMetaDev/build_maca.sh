#!/usr/bin/env bash
# ============================================================
# build_maca.sh —— 沐曦（MetaX / 曦云 C500）平台构建
#   用法: bash build_maca.sh
#   产物: ./build_maca/lowprec
#
#   与 build.sh（nvcc + CMake）的区别：
#     1. 沐曦用 mxcc 编译，设备代码带 -x maca、架构用 -offload-arch native；
#     2. CUDA 兼容头文件在 $MACA/tools/cu-bridge/include（把 cudaXxx 声明为
#        wcudaXxx，实现在 $MACA/lib/libruntime_cu.so），必须按 MACA 的
#        cu-bridge/bin/conf.json 里 [link][adder] 追加链接参数；
#     3. -imacros __macro_mxcc.h 让 __CUDACC__ 等 CUDA 宏在 mxcc 下生效；
#     4. fast-math 开关是 -use-fast-math（连字符，非 nvcc 的 -use_fast_math）。
#
#   本平台的 warpSize = 64（不是 32），而本工程用的是
#   __shfl_xor_sync(mask, v, off, WIDTH)（WIDTH ≤ 32）的段内规约，
#   属于需要实测确认的点 —— 由 tests/run_tests.sh 的"CPU 参考 vs GPU 逐位一致"覆盖。
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
echo "[build_maca] 编译 lowprec（5 个源文件，一次编译链接）"
# 注意：本工程刻意不加 -use-fast-math。
# 实测在 2048x2048 的 random 数据上，开 fast-math 会让 nvfp4 block 模式的
# 块缩放推导在临界块上出现 1 ULP 级偏差（16 个元素 = 恰好一个 block 不一致），
# 而本工程要求"CPU 参考 vs GPU 逐位一致"，因此必须与 nvcc 默认（不带 fast-math）对齐。
"$MXCC" -x maca -offload-arch native --maca-path="$MACA" \
    -I"$ROOT/include" -I"$BRIDGE_INC" -L"$MACA/lib" \
    -imacros __macro_mxcc.h -forward-unknown-to-compiler \
    -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    -Wno-unused-command-line-argument \
    -std=c++17 -O3 \
    src/main.cpp src/io.cpp src/metrics.cpp src/reference.cpp src/kernels.cu \
    -o "$BUILD_DIR/lowprec"

echo "构建完成: $ROOT/$BUILD_DIR/lowprec"
echo "测试: BIN=./$BUILD_DIR/lowprec bash tests/run_tests.sh"
