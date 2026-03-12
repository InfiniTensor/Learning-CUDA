#!/bin/bash
# run_all.sh

# 确保 CUDA 程序已编译
if [ ! -f "./nf4_dequant_cuda" ]; then
    echo "编译 CUDA 程序..."
    nvcc -O3 -arch=sm_90 --use_fast_math -lineinfo -Xptxas -O3 nf4_dequant_cuda.cu -o nf4_dequant_cuda
fi

# 处理所有权重文件
for f in ../weight_data/*.bin; do
     ./nf4_dequant_cuda "$f"
done