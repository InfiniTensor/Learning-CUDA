#!/bin/bash
set -euo pipefail

if [ ! -f "./nf4_dequant_mx" ]; then
  echo "编译沐曦 NF4 程序..."
  if command -v mxcc >/dev/null 2>&1; then
    mxcc -O3 -std=c++17 -o nf4_dequant_mx nf4_dequant_mx.cu
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc -O3 -std=c++17 -o nf4_dequant_mx nf4_dequant_mx.cu \
      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
  else
    echo "未找到 mxcc/nvcc，请先加载沐曦编译环境。"
    exit 1
  fi
fi

for f in ../weight_data/*.bin; do
  ./nf4_dequant_mx "$f"
done
