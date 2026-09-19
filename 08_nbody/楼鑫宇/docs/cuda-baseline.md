# CUDA 直接求和基线与分块实验

## 实验范围

- 日期：2026-09-14
- GPU：NVIDIA GeForce RTX 5090，compute capability 12.0
- CUDA Toolkit：12.8.61
- 构建：Release，`CMAKE_CUDA_ARCHITECTURES=120`
- 精度：CUDA FP32；CPU 参考为 FP64
- 积分器：Leapfrog KDK
- 输入：固定种子 42 的星团
- 每个 GPU 配置独立运行 5 次，表中报告模拟时间的最小值、中位数和最大值

4096 粒子实验运行 1000 步、每 10 步记录一次。16384 粒子调参实验运行 100 步，只记录初态和末态。`simulation_wall_ms` 包含模拟期间的轨迹回传，不包含输入解析、CUDA 初始化和末尾文件写入。

## 正确性与内存检查

129 粒子场景用于覆盖 block size 为 64、128、256、512 时的尾块。每个 CUDA 后端逐粒子比较 CPU FP64 的最终位置、速度和全部记录轨迹。

| 指标 | 最大绝对误差 |
|---|---:|
| naive 最终位置 | 7.904e-8 |
| tiled 最终位置 | 7.904e-8 |
| tiled 最终速度 | 8.462e-8 |
| tiled 轨迹 | 8.941e-8 |

Compute Sanitizer 的 `memcheck` 和 `synccheck` 均报告 0 个错误。项目全部五项 CTest 测试通过。

## 4096 粒子、1000 步

| 后端 | block size | 力计算中位数 ms | 模拟最小 ms | 模拟中位数 ms | 模拟最大 ms |
|---|---:|---:|---:|---:|---:|
| cuda-naive | 128 | 210.069 | 222.291 | 222.421 | 222.446 |
| cuda-tiled | 64 | 216.258 | 227.230 | 227.294 | 227.595 |
| cuda-tiled | 128 | 209.685 | 220.929 | 221.005 | 221.059 |
| cuda-tiled | 256 | 214.153 | 225.679 | 225.751 | 228.179 |
| cuda-tiled | 512 | 239.305 | 251.869 | 251.939 | 252.371 |

同一输入的 CPU FP64 单次基线为 67,608.592 ms。因此 tiled B=128 相对 CPU 的模拟加速比约为 305.9 倍。tiled B=128 相对 naive B=128 仅快约 0.64%。两种 CUDA 后端的相对能量误差均为 2.288e-6，CPU FP64 为 2.307e-6。

## 16384 粒子、100 步调参

| 后端 | block size | 力计算中位数 ms | 模拟最小 ms | 模拟中位数 ms | 模拟最大 ms |
|---|---:|---:|---:|---:|---:|
| cuda-naive | 128 | 84.248 | 85.320 | 85.351 | 85.383 |
| cuda-tiled | 64 | 86.228 | 87.338 | 87.343 | 87.379 |
| cuda-tiled | 128 | 83.577 | 84.687 | 84.695 | 84.701 |
| cuda-tiled | 256 | 85.483 | 86.588 | 86.609 | 86.627 |
| cuda-tiled | 512 | 95.222 | 96.383 | 96.395 | 96.418 |

B=128 仍是最优配置；tiled 相对 naive 快约 0.77%。在 RTX 5090 上，朴素内核对连续源粒子的读取已经得到缓存和广播机制的良好服务，而 tiled 每个 tile 增加两次块同步，因此共享内存优化的净收益很小。该结果应保留为实测结论，不能表述成显著加速。

## 复现命令

```bash
source /data/nbody-dev/activate-nbody.sh
cd /data/nbody-project/08_nbody/楼鑫宇

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
ctest --test-dir build --output-on-failure

compute-sanitizer --tool memcheck ./build/nbody_cuda_tests
compute-sanitizer --tool synccheck ./build/nbody_cuda_tests

./build/nbody --backend cuda-tiled --block-size 128 \
  --input data/cluster_4096.txt \
  --config configs/cpu_4096.cfg \
  --output results/cluster_4096_cuda_tiled
```

16384 粒子调参使用 `configs/benchmark_100.cfg`。原始运行目录保存在服务器 `results/` 下，该目录按项目约定不提交 Git；正式报告使用这些 JSON 原始值生成表格，不手工估造数据。
