#!/bin/bash
# run_all.sh

# 确保 CUDA 程序已编译
if [ ! -f "./nf4_dequant_cuda" ]; then
    echo "编译 CUDA 程序..."
    nvcc -O3 -o nf4_dequant_cuda nf4_dequant_cuda.cu \
        -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 -lcudart
fi

# 处理所有权重文件
for f in ../weight_data/*.bin; do
    ./nf4_dequant_cuda "$f"
done