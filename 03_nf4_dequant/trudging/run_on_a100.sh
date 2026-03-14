#!/bin/bash
#SBATCH --job-name=nf4_dequant              # 任务名
#SBATCH --output=result_%j.log              # 标准输出文件
#SBATCH --error=error_%j.log                # 标准错误输出文件
#SBATCH --partition=nvidia                  # 分区名
#SBATCH --nodes=1                           # 节点数
#SBATCH --ntasks=1                          # 总任务数
#SBATCH --cpus-per-task=16                  # 每个任务需要的 CPU 核心数
#SBATCH --gres=gpu:nvidia:1                 # 请求 1 块 A100 GPU (对应测试即可)
#SBATCH --mem=64G                           # 请求的内存
#SBATCH --time=00:10:00                     # 运行时间上限 (10分钟足够)

# 1. 设置 CUDA 环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "============ Starting Compilation ============"
# 使用 nvcc 编译代码。平台为 A100，固定架构为 sm_80
nvcc -O3 -lineinfo --ptxas-options=-v -use_fast_math -arch=sm_80 main.cu src/dequantize.cu -o nf4_dequantizer

if [ $? -eq 0 ]; then
    echo "============ Compilation Success ============"
    echo "============ Running Kernel ============"
    # 2. 运行算子
    srun ./nf4_dequantizer
else
    echo "============ Compilation Failed ============"
fi
