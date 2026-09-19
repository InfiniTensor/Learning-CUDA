# N 体引力模拟与可视化（CUDA）

2026 夏季训练营 · 选题八 · `08_nbody/jiaying`。

在 WSL2 / Ubuntu / NVIDIA GPU 上运行的完整项目：CUDA 直接引力计算、leapfrog 与 Euler 积分、CPU 对照、二进制轨迹、NumPy + Matplotlib 动画、正确性验证和性能实验。

实测 RTX 4070 Laptop GPU：4096 粒子 × 1000 步计算耗时 **0.256 秒**，相对 CPU 单线程计算加速 **144.8 倍**；包含启动、读写、守恒检查的完整进程加速 **65.5 倍**。65536 粒子 × 1000 步计算耗时 **13.986 秒**。详见 [实验报告](REPORT.md)、[原始日志](reports/benchmark.csv) 和 [动画画廊](reports/gallery.html)。这些是本机实测，其他硬件和功耗状态的结果会不同。

## 1. WSL 快速开始

在 WSL 终端执行：

```bash
cd /mnt/d/Jiaying/Documents/ChatGPT/InfiniTensor/Learning-CUDA/08_nbody/jiaying
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash scripts/build_wsl.sh -DCMAKE_CUDA_ARCHITECTURES=89
bash scripts/demo_wsl.sh
```

构建脚本会执行正确性验证。`demo_wsl.sh` 生成 `output/cluster.gif` 和 `output/binary.gif`，可直接从 Windows 打开。图形桌面可用时，省略 `--save` 可以打开交互动画窗口。

依赖：CMake ≥ 3.24、支持 C++17 的 GCC、CUDA Toolkit、OpenMP、Python、NumPy、Matplotlib、Pillow。本机使用 CUDA 12.4 + GCC 13；CUDA 12.4 不宜直接使用 Ubuntu 最新的默认 GCC。构建时可用 `CXX=g++-13` 指定兼容版本。

```bash
# 仅在依赖缺失时安装；CUDA Toolkit 须预先安装并可运行 nvidia-smi。
sudo apt-get update
sudo apt-get install g++-13 cmake python3-numpy python3-matplotlib python3-pil
```

本机已安装可视化依赖。默认 CMake 构建目标为 SM 75、86、89；`-DCMAKE_CUDA_ARCHITECTURES=89` 用于本机 Ada GPU。其他 NVIDIA GPU 请设置对应架构。项目只依赖通用 CUDA 核函数，不使用 Tensor Core。

## 2. 模拟输入和命令

粒子文件每行 `x y z vx vy vz mass`，接受空行与 `#` 注释，质量必须为正。配置示例：

```text
dt = 0.001
num_steps = 1000
record_interval = 10
G = 1.0
softening = 0.03
integrator = "leapfrog"
```

```bash
python3 scripts/generate.py data/cluster.txt --particles 4096 --seed 2026
build/nbody --input data/cluster.txt --config configs/cluster.txt \
  --output output/cluster.bin --log output/cluster.json --final output/final.txt

# 相同工作量的 CPU 对照，不记录轨迹，输出速度比和末态位置误差。
build/nbody --input data/cluster.txt --config configs/cluster.txt \
  --no-trajectory --compare-cpu --log output/comparison.json

# CPU 后端默认单线程，也可用 --cpu-threads 12。
build/nbody --input data/cluster.txt --config configs/cluster.txt \
  --backend cpu --output output/cpu.bin --log output/cpu.json
```

参数：

| 参数 | 含义 |
|---|---|
| `--backend cuda/cpu` | 默认 CUDA，CPU 单线程可作基准 |
| `--kernel tiled/naive` | 默认共享内存分块；naive 用于优化对照 |
| `--block-size 128/256` | 默认 128 |
| `--cpu-threads N` | 默认 1，CPU OpenMP 线程数 |
| `--no-trajectory` | 关闭轨迹传输和写入 |
| `--compare-cpu` | 附加同初态 CPU 运行，需同时指定 `--no-trajectory` |
| `--no-diagnostics` | 跳过 CPU FP64 全量 O(N²) 能量检查，大规模运行建议使用 |
| `--final 文件` | 保存末态七列数据，可作为下一次模拟的输入 |
| `--help` | 完整命令说明 |

`num_steps=0` 支持只记录初态。`record_interval` 大于步数时仍记录初态与末态。Euler 使用显式 Euler；默认 leapfrog 使用 kick–drift–kick，输出速度为整步速度。软化参数只控制引力势的平滑，不是粒子半径。

## 3. 二进制格式

完全采用题目要求的**粒子优先顺序**，无额外二进制字段：

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| P | 小端 int32 | 粒子数 |
| R | 小端 int32 | 记录点数 |
| positions | 小端 float32 `[P, R, 3]` | 同一粒子的全部 `x,y,z`，然后下一个粒子 |

文件长度 `8 + 12*P*R` 字节。记录 step=0、每个间隔和末步；末步不会重复记录。旁边的 `*.bin.json` 提供真实步号和 dt。缺少旁文件时可视化以记录序号作为横向时间标注。

```python
from scripts.trajectory import load_trajectory
positions, times, metadata = load_trajectory("output/cluster.bin")
# positions[particle_id, record_index, xyz]，使用 NumPy memmap 按需读取。
```

C++ 通过 Linux mmap 直接写最终的粒子优先布局，不需要把全部帧保留在显存或额外构造完整转置数组。物理内存/page cache 仍可能随着输出文件增大；结束时 `msync` 的耗时单独记录。

## 4. 动画和场景

```bash
python3 visualize.py output/cluster.bin --save output/cluster.gif
python3 visualize.py output/cluster.bin --dimensions 3 --save output/cluster3d.gif
python3 visualize.py output/cluster.bin --save output/cluster.html
python3 visualize.py output/cluster.bin --risk-radius 0.05 --snapshot output/cluster.png
```

支持 GIF、HTML、MP4（MP4 另需 ffmpeg），实现使用 `matplotlib.animation.FuncAnimation`。`--stride` 控制抽帧，`--trail` 控制轨迹尾长，`--max-particles` 只限制画出的粒子数量。显示范围覆盖画出的全部轨迹。

初态生成器支持 `cluster`（截断 Plummer 星团）、`binary`（圆轨道）、`ellipse`（椭圆状轨道）、`ring`（中心星体、小天体环和扰动天体）、`collision`（相向运动的两个星团）。二体场景固定 2 个粒子。

`--risk-radius` 以三维真实距离在**全部粒子**中统计每帧近距离配对，空间哈希避免通常情况下的全量两两检查，并用红色标出涉及的粒子。它是离散快照的近距离提示，既不模拟接触/合并，也不检测两个记录时刻之间可能发生的穿越。最坏情况下（粒子集中在同一网格）仍可能 O(N²)，可增大 `--stride` 或关闭该选项。

已有动画和图表在 `reports/`；运行完整 benchmark 后可统一重建：

```bash
bash scripts/render_demos_wsl.sh
python3 scripts/analyze.py
```

## 5. 复现实验

```bash
ctest --test-dir build --output-on-failure
python3 scripts/benchmark.py
bash scripts/profile_wsl.sh
bash scripts/sanitize_wsl.sh
```

benchmark 会生成 33 条记录：3 个规模 × 3 个 CUDA 配置 × 3 次重复、4096×1000 的 CUDA/CPU 完整流程、65536×1000，以及 CPU 多线程对照。默认运行全部工作量，CPU 完整测试本机约 37 秒。`analyze.py` 校验完整轨迹，并生成性能/轨道误差图。

本机限制：Compute Sanitizer 因 WDDM 调试接口未开放而无法执行；已保存失败原文，**不能视为 memcheck/racecheck 通过**。Nsight Systems 2023.4 获得了 CUDA API 记录，但没有 GPU 内核/显存活动数据。数值验证、完整规模试验和主机 UBSan 验证结果见报告。

## 6. 提交

源代码、脚本、独立验证文件、报告及实测结果均位于 `08_nbody/jiaying/`。按题目要求提交到 Learning-CUDA 的 `2026-summer-project` 分支；提交范围只选择本目录。`build*/`、`data/`、`output/` 被忽略，避免提交可执行文件和大轨迹。

```bash
python3 scripts/package.py
```

该命令在当前项目目录生成可复核的提交压缩包和 SHA-256 清单；它不替代官网要求的 fork/commit/PR 链接。
