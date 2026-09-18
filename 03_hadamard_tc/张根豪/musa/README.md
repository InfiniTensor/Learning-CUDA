# FastHadamard-MUSA

这是 `FastHadamard-CUDA` 的摩尔线程 MUSA 国产平台适配子目录。`src/` 中的
`.mu` 文件基于 CUDA 实现复制并适配，算法、数据布局和 CPU 参考结果保持一致，
使用 MUSA toolkit 的 CUDA 兼容头和 `mcc` 编译。CUDA 主工程的默认构建不依赖本目录。

## 构建

```bash
source env.sh
make -j2
```

如果工具链不在默认位置，可传入 `MUSA_HOME`。构建使用 Ninja 和 MUSA 官方
CMake language module，不调用原工程的 `nvcc`/CUDA CMake 配置。

## S4000 冒烟验证

```bash
source env.sh
make smoke
```

更大规模的正确性测试：

```bash
LD_LIBRARY_PATH=/usr/local/musa-5.1.0/lib:$LD_LIBRARY_PATH \
./build/hadamard_bench --batch 1 --seq 128 --heads 8 --head_dim 128 \
  --dtype fp16 --normalize true --warmup 3 --iters 10 --check true
```

## INT4 与性能验收

高级基准同时验证：优化核与基线 bit-exact、融合 INT4 与先变换后量化
bit-exact，以及 GPU/CPU INT4 结果的 scale 和反量化误差。`--csv` 会保存
每个实现的执行时间、加速比和量化误差：

```bash
./build/hadamard_advanced_bench --batch 1 --seq 128 --heads 8 \
  --head_dim 256 --dtype bf16 --warmup 20 --iters 100 \
  --csv results/musa_advanced.csv
```

MUSA runtime 支持用 `--profile true` 标记需要采集的计时区间：

```bash
./build/hadamard_advanced_bench --batch 1 --seq 128 --heads 8 \
  --head_dim 256 --dtype bf16 --warmup 20 --iters 100 \
  --profile true --csv results/musa_profile.csv
```

本适配环境需要额外安装 MUSA profiler 命令行工具才能生成硬件计数器报告；
当前镜像已验证 runtime profiler API 和性能计时，但未安装 `ncu`、`nsys` 或
`msprof`，因此不能伪造 profiler 指标。