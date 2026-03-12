#!/bin/bash
#SBATCH --job-name=test_job              # 任务名
#SBATCH --output=output_%j.log           # 标准输出文件（%j 会替换成 job ID）
#SBATCH --error=error_%j.log             # 标准错误输出文件
#SBATCH --partition=mt                   # 分区名（机器系统默认分区是 mt）
#SBATCH --nodes=1                        # 需要的节点数
#SBATCH --ntasks=1                       # 总任务数（通常 = 节点数 × 每节点任务数）
#SBATCH --cpus-per-task=16               # 每个任务需要的 CPU 核心数
#SBATCH --gres=gpu:mt:1                  # 请求 4 块 GPU（mt 是 Gres 类型）
#SBATCH --mem=256G                       # 请求的内存
#SBATCH --time=00:20:00                  # 运行时间上限





for f in ../weight_data/*.bin; do
    srun ./nf4_musa "$f"
done