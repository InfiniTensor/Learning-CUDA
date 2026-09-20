# N 体引力模拟与可视化（CUDA）

这是一个可编译、运行、测试、profiling 和可视化的 CUDA direct N-body 项目。GPU 使用 SoA 存储位置、速度和质量，引力计算复杂度为 `O(N²)`；积分支持 Euler 与严格 kick-drift-kick Leapfrog。轨迹按 `record_interval` 写入紧凑二进制文件，性能日志区分 CUDA Event kernel 时间、传输、磁盘写入和总 wall time。

## 项目结构

```text
nbody_cuda/
├── CMakeLists.txt
├── README.md
├── OPTIMIZATION.md          # baseline 与 shared-memory 阶段记录
├── PROFILING.md             # 最终 nsys/ncu、sweep、正确性报告
├── src/                     # CUDA、CPU reference、I/O 和主程序
├── scripts/                 # 数据生成、benchmark、可视化
├── tests/                   # CTest 输入与 smoke test
└── results/                 # 真实 CSV 和 profiler 报告
```

## 四种 force 路径

| `--kernel` | 做法 | 适合观察的问题 |
|---|---|---|
| `baseline` | 一个 thread 负责一个 target，直接遍历全部 source | 缓存复用与基础性能 |
| `tiled` | block 把 `x/y/z/mass` 分块搬进 shared memory | shared reuse 与 barrier 成本 |
| `tiled-float4` | shared tiling + `float4(x,y,z,mass)` + 小幅展开 | 大 N 下 L1/TEX 压力 |
| `split-source` | source 维拆为多个 chunk，partial kernel 后接 reduction | 小 N target grid 过小 |

所有路径都把 `ax/ay/az` 放在寄存器中累加，距离使用带 softening 的 `rsqrtf`：

```text
dist2 = dx*dx + dy*dy + dz*dz + softening*softening
inv_r = rsqrtf(dist2)
scale = G * mass * inv_r * inv_r * inv_r
```

Leapfrog 初始化时先计算 `a(t0)`，每步执行 `kick_drift → a(t+dt) → final_kick`。新加速度直接留给下一步使用；200 步共有 201 次 force evaluation。普通路径为 601 个 kernel，split-source 为 802 个 kernel。

## 依赖与构建

需要 NVIDIA GPU/驱动、CUDA Toolkit、CMake 3.20+、C++17 编译器，以及用于脚本的 Python 3、NumPy、Matplotlib。

Linux / WSL2 Release：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows Release：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

NVTX 默认关闭；nsys 使用单独构建目录：

```bash
cmake -S . -B build-nvtx -DENABLE_NVTX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-nvtx -j
```

可选构建开关还有 `NBODY_FAST_MATH=ON` 与 `NBODY_PTXAS_INFO=ON`。fast math 默认关闭，避免把精度变化混入标准结果。默认 CUDA 架构为 75 和 86，可通过 `-DCMAKE_CUDA_ARCHITECTURES=89` 等参数覆盖。

## 运行与输入输出

```bash
./build/nbody tests/two_body.txt tests/two_body.cfg \
  --kernel tiled-float4 --block-size 128 \
  --output results/two_body.bin \
  --performance-log results/two_body.log
```

Windows 将可执行文件换成 `.\build\Release\nbody.exe`。完整选项可运行 `nbody --help`；主要参数为：

```text
--kernel baseline|tiled|tiled-float4|split-source
--block-size N
--source-chunks N
--profile-steps N
--cuda-graph
--skip-conservation
--cpu-validation auto|on|off
--output PATH
--performance-log PATH
```

`--cuda-graph` 默认关闭，capture 一个固定 timestep；trajectory D2H 和文件写入仍位于 Graph 外并遵守 `record_interval`。`--profile-steps` 用于缩短 ncu replay；`--skip-conservation` 避免大 N 时 CPU 的 `O(N²)` 守恒量统计干扰 profiler。

粒子文件每行是 `x y z vx vy vz mass`。配置示例：

```ini
dt = 1e-3
num_steps = 200
record_interval = 20
G = 1.0
softening = 1e-4
integrator = leapfrog
kernel = split-source
block_size = 64
source_chunks = 32
cuda_graph = false
```

轨迹格式为两个 `int32` 头字段，后接 `float32 positions[record_count][N][3]`。初始帧总会记录，后续仅在 `step % record_interval == 0` 时 D2H：

```text
record_count = 1 + floor(num_steps / record_interval)
```

## 正确性与可视化

CTest 覆盖四种 force 路径、split-source Graph、Euler/Leapfrog、CPU reference、轨迹一致性与两体轨道。split-source 改变浮点加法分组，因此按测试容限比较，不要求 bitwise identical。

```bash
ctest --test-dir build --output-on-failure
python scripts/visualize.py results/two_body.bin --info
python scripts/visualize.py results/two_body.bin
python scripts/visualize.py results/two_body.bin --3d --save orbit.gif
```

## 性能 sweep

`benchmark.py` 保存每次原始数据到 `benchmark.csv`，中位数汇总到 `summary.csv`。汇总字段包含 N、kernel、block size、source chunks、Graph、steps、GPU/wall 中位数及 particle-steps/s；失败或变慢的结果不会被过滤。

source chunk sweep：

```bash
python scripts/benchmark.py ./build/nbody \
  --sizes 4096 8192 16384 --steps 200 --adaptive-steps \
  --kernel split-source --block-sizes 64 \
  --source-chunks 1 2 4 8 16 32 64 --repeats 3 \
  --output-dir results/final_chunk_sweep
```

最终 N sweep，包括 split-source Graph on/off：

```bash
python scripts/benchmark.py ./build/nbody \
  --sizes 1024 2048 4096 8192 16384 32768 65536 \
  --steps 200 --adaptive-steps \
  --kernels baseline tiled tiled-float4 split-source --block-sizes 64 \
  --chunks-by-size 1024:64 2048:64 4096:32 8192:16 \
                   16384:8 32768:4 65536:2 \
  --include-split-graph --repeats 5 \
  --output-dir results/final_n_sweep
```

## Profiler

Nsight Systems 用于看 launch、GPU timeline、copy 和同步：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o results/split_timeline ./build-nvtx/nbody particles.txt config.txt \
  --kernel split-source --source-chunks 32 --profile-steps 200 \
  --cpu-validation off --skip-conservation
```

Nsight Compute 只抓少量 force launch，避免 `--set full` 对数百步反复 replay：

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section SchedulerStats --section WarpStateStats \
  --section LaunchStats --kernel-name regex:compute_acceleration_baseline \
  --launch-skip 5 --launch-count 1 ./build/nbody particles.txt config.txt \
  --kernel baseline --profile-steps 10 --cpu-validation off --skip-conservation
```

重点看 duration、SM/DRAM/L1/L2 throughput、cache hit、global/shared load、achieved occupancy、registers/thread、eligible warps 与 stall reasons。split-source 可把 kernel regex 换成 `compute_acceleration_partial`。完整实测、Graph 证据、chunk/N sweep 与解释见 [PROFILING.md](PROFILING.md)。
