#!/usr/bin/env bash
# ============================================================================
# 天数智芯（Iluvatar CoreX）平台构建脚本
#   用法: bash build_iluvatar.sh
#   依赖: IX-ML / CoreX SDK（默认 /usr/local/corex-4.4.0，可用 COREX_HOME 覆盖）
#
#   与 build.sh（NVIDIA/nvcc）的区别：
#     1. 天数用 clang++ 编译，且 .cu 文件必须加 -x ivcore；
#     2. 源码头文件来自 $COREX/include，运行时库来自 $COREX/lib64；
#     3. 定义 PLATFORM_ILUVATAR 以启用平台适配代码（见 src/gpu_maxflow.cu
#        中 64 位原子加的实现选择）；
#     4. 运行可执行文件前需要 LD_LIBRARY_PATH=$COREX/lib64。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 自动探测 CoreX SDK ----------
COREX="${COREX_HOME:-}"
if [ -z "$COREX" ]; then
    for d in /usr/local/corex-4.4.0 /usr/local/corex /usr/local/ix* ; do
        if [ -x "$d/bin/clang++" ]; then
            COREX="$d"
            break
        fi
    done
fi
if [ -z "$COREX" ]; then
    echo "ERROR: 未找到天数 CoreX SDK（需要 \$COREX/bin/clang++）" >&2
    echo "       可通过 COREX_HOME=/path/to/corex 指定" >&2
    exit 2
fi

CXX="$COREX/bin/clang++"
ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build_iluvatar}"
INC="-I$ROOT/include -I$COREX/include"
LIBS="-L$COREX/lib64 -lcudart"
FLAGS="-std=c++17 -O3 -DPLATFORM_ILUVATAR"

export LD_LIBRARY_PATH="$COREX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[build_iluvatar] CoreX SDK   = $COREX"
echo "[build_iluvatar] 编译器      = $("$CXX" --version | head -1)"
if command -v ixsmi >/dev/null 2>&1 || [ -x "$COREX/bin/ixsmi" ]; then
    "$COREX/bin/ixsmi" 2>/dev/null | sed -n '4,7p' | sed 's/^/[build_iluvatar] /' || true
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[build_iluvatar] 编译设备代码 src/gpu_maxflow.cu（-x ivcore）"
"$CXX" -x ivcore $FLAGS $INC -c ../src/gpu_maxflow.cu -o gpu_maxflow.o

echo "[build_iluvatar] 编译主机代码"
for f in main io cpu_ref; do
    "$CXX" $FLAGS $INC -c "../src/$f.cpp" -o "$f.o"
done

echo "[build_iluvatar] 链接"
"$CXX" ./*.o $LIBS -o maxflow

echo "构建完成: $(pwd)/maxflow"
echo "运行前请设置: export LD_LIBRARY_PATH=$COREX/lib64:\$LD_LIBRARY_PATH"
