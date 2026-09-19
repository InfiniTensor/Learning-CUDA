# M5 性能分析与可视化实验记录

本文件记录 M5 阶段的原始实验条件、关键结果和复现命令，不作为最终项目报告。测试日期为 2026-09-14，设备为 NVIDIA GeForce RTX 5090，后端为 `cuda-tiled`，block size 为 128。

## 1. 轨迹输出开销

固定初态 `data/cluster_4096.txt`，执行 4096 粒子、1000 步，每种记录策略各启动 5 个独立进程并串行测量。模拟时间来自程序内部计时，端到端时间包含进程启动、CUDA 上下文初始化和结果写盘。

| 记录策略 | 帧间隔 K | 模拟时间中位数 / ms | 冷启动端到端中位数 / ms | 轨迹大小 |
| --- | ---: | ---: | ---: | ---: |
| 关闭轨迹 | - | 218.246 | 542.123 | 0 B |
| 每步记录 | 1 | 246.267 | 930.576 | 49,201,160 B |
| 每 10 步记录 | 10 | 221.141 | 552.392 | 4,964,360 B |
| 每 100 步记录 | 100 | 218.693 | 508.827 | 540,680 B |

逐步记录使模拟段相对关闭记录增加约 12.8%，端到端时间增加约 71.7%。K=10 的模拟段增加约 1.3%，已能把轨迹体积缩小到 K=1 的约十分之一；K=100 的模拟时间基本回到关闭记录的水平。完整汇总数据在 [`io-cost-summary.csv`](io-cost-summary.csv)，图表在 [`io-cost.png`](io-cost.png)。每轮 stdout、stderr、`performance.json` 与完整 `raw.csv` 保存在服务器的 `results/io_cost_m5/`。

复现：

```bash
python scripts/benchmark_io.py --output-root results/io_cost_m5 --repeat 5 --block-size 128
python scripts/plot_io_results.py results/io_cost_m5/summary.csv \
  --output results/io_cost_m5/io_cost.png
```

## 2. Nsight Systems 时间线

用 4096 粒子、20 步短配置并关闭轨迹与守恒量计算，避免 I/O 干扰 kernel 时间线：

```bash
nsys profile --trace=cuda,osrt --sample=none \
  -o results/profiles/m5_timeline \
  ./build/nbody --backend cuda-tiled --block-size 128 \
  --record off --diagnostics off \
  --input data/cluster_4096.txt --config configs/profile.cfg \
  --output results/profiles/m5_profile_run

nsys stats results/profiles/m5_timeline.nsys-rep
```

统计中共有 61 次 kernel launch。21 次 `force_tiled_kernel` 总计 4.321 ms，平均约 205.8 us，占 GPU kernel 时间的 99.2%；20 次 kick-drift 与 20 次 final-kick 各约 0.8%。力计算是明确主瓶颈，积分 kernel 融合目前不会显著改变计算段总耗时。该短进程中三次 `cudaMalloc` API 共 132.4 ms，主要是首次 CUDA 上下文初始化，因此正式性能采样应使用重复运行并区分模拟段与冷启动端到端时间。

报告文件保存在服务器的 `results/profiles/m5_timeline.nsys-rep`。尝试用 Nsight Compute 收集硬件计数器时，平台返回 `ERR_NVGPUCTRPERM`；这是服务器计数器权限限制，不是 kernel 失败，应用本身正常完成。

## 3. 星团与小天体带可视化

星团动画来自 4096 粒子 tiled 运行，输出为：

```text
results/cluster_4096_cuda_tiled_b128_r1/cluster_tiled_xy.mp4
```

小天体实验以质量 1000 的中心天体和 512 个总质量为 1 的环带粒子为初态，半径范围为 5 到 10，随机种子为 42。扰动组额外加入质量 1、初始位置 `(8, 0, 1)`、切向速度为当地圆轨道速度 0.8 倍的扰动体。两组均运行 5000 步、每 10 步记录一次：

```bash
python scripts/generate_asteroid.py --output data/asteroid_512_control.txt
python scripts/generate_asteroid.py --perturber \
  --output data/asteroid_512_perturbed.txt

./build/nbody --backend cuda-tiled --block-size 128 \
  --input data/asteroid_512_control.txt --config configs/asteroid.cfg \
  --output results/asteroid_control_cuda_tiled
./build/nbody --backend cuda-tiled --block-size 128 \
  --input data/asteroid_512_perturbed.txt --config configs/asteroid.cfg \
  --output results/asteroid_perturbed_cuda_tiled
```

两组轨迹均全部为有限值。按中心天体对齐后，扰动组与对照组对应小天体的末态位移差中位数为 0.422，最大值为 11.431；相对能量误差分别为 `1.87e-7` 和 `1.49e-6`。动画位于：

```text
results/asteroid_control_cuda_tiled/asteroid_control_xy.mp4
results/asteroid_perturbed_cuda_tiled/asteroid_perturbed_xy.mp4
```

可重新渲染为：

```bash
python scripts/visualize.py results/asteroid_control_cuda_tiled \
  --output results/asteroid_control_cuda_tiled/asteroid_control_xy.mp4 \
  --max-particles 513 --title "Asteroid belt: control"
python scripts/visualize.py results/asteroid_perturbed_cuda_tiled \
  --output results/asteroid_perturbed_cuda_tiled/asteroid_perturbed_xy.mp4 \
  --max-particles 514 --title "Asteroid belt: perturbed"
```

## 4. 本阶段结论

- `cuda-tiled` 的主要耗时仍在 O(N²) 力计算 kernel，当前积分 kernel 启动开销不是首要优化对象。
- 常规动画使用 K=10 是较好的体积与时间折中；性能基准应使用 `--record off --diagnostics off`。
- 扰动体已产生清晰且可量化的轨道差异，同时数值结果保持有限、能量漂移较小。
