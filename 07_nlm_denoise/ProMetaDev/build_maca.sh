#!/usr/bin/env bash
# ============================================================
# build_maca.sh —— 沐曦（MetaX / 曦云 C500）平台构建
#   用法: bash build_maca.sh
#   产物: ./build_maca/nlm
#
#   与 build.sh（nvcc + CMake）的区别：
#     1. 沐曦用 mxcc 编译，设备代码带 -x maca、架构用 -offload-arch native；
#     2. CUDA 兼容头文件在 $MACA/tools/cu-bridge/include（把 cudaXxx 声明为
#        wcudaXxx，实现在 $MACA/lib/libruntime_cu.so），必须按 MACA 的
#        cu-bridge/bin/conf.json 里 [link][adder] 追加链接参数；
#     3. -imacros __macro_mxcc.h 让 __CUDACC__ 等 CUDA 宏在 mxcc 下生效；
#     4. 本工程依赖 OpenCV（图像读写 + fastNlMeansDenoising 参考实现），
#        容器里需先装 libopencv-dev；头文件与库参数由 pkg-config 自动取。
#        沐曦容器实测可直接 `apt-get install -y --no-install-recommends libopencv-dev`
#        （源可达），另建议装 python3-numpy / python3-pil 供 tests 里的
#        gen_image.py 与 PSNR 计算使用。
#     5. 本工程不加 -use-fast-math（与工程 CMake 的 nvcc 默认行为保持一致，
#        避免影响 tests 里 MAE / PSNR 的阈值判定）。
#
#   测试脚本 tests/run_tests.sh 固定用 ./build/nlm，
#   可先 `mkdir -p build && cp build_maca/nlm build/` 再跑。
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

if ! pkg-config --exists opencv4 2>/dev/null; then
    echo "ERROR: 未找到 OpenCV（请先 apt-get install -y libopencv-dev）" >&2
    exit 2
fi
OCV_CFLAGS="$(pkg-config --cflags opencv4)"
OCV_LIBS="$(pkg-config --libs opencv4)"

export LD_LIBRARY_PATH="$MACA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "---------------------------------------------"
echo "[build_maca] MACA SDK = $MACA"
echo "[build_maca] mxcc     = $("$MXCC" --version 2>&1 | head -1)"
echo "[build_maca] OpenCV   = $(pkg-config --modversion opencv4)"
command -v mx-smi >/dev/null 2>&1 && echo "[build_maca] 设备     = $(mx-smi 2>/dev/null | grep -oE 'MetaX [A-Za-z0-9]+' | head -1)"
echo "---------------------------------------------"

mkdir -p "$BUILD_DIR"
echo "[build_maca] 编译 nlm（4 个源文件 + OpenCV，一次编译链接）"
"$MXCC" -x maca -offload-arch native --maca-path="$MACA" \
    -I"$ROOT/include" -I"$BRIDGE_INC" -L"$MACA/lib" \
    $OCV_CFLAGS \
    -imacros __macro_mxcc.h -forward-unknown-to-compiler \
    -fgpu-rdc --maca-link -lToolsExt_cu -lruntime_cu -lmcToolsExt \
    -Wno-unused-command-line-argument \
    -std=c++17 -O3 \
    src/main.cpp src/image_io.cpp src/cpu_ref.cpp src/nlm_gpu.cu \
    $OCV_LIBS \
    -o "$BUILD_DIR/nlm"

echo "构建完成: $ROOT/$BUILD_DIR/nlm"
echo "测试: mkdir -p build && cp $BUILD_DIR/nlm build/ && bash tests/run_tests.sh"
