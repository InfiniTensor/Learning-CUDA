# 第二阶段性能优化记录

> 这份文档保留第二阶段当时的 baseline 与 shared-memory 结论。最终的 NVTX、nsys/ncu、split-source、CUDA Graph 和规模 sweep 实测见 `PROFILING.md`。

这份记录对应当前代码，不把理论上的访存减少直接当成真实性能提升。所有表格数据来自 RTX 3080、CUDA 12.8 的实际运行，中位数由三次重复得到。

## 1. 修改前的实现

### CUDA 并行方式

baseline 和 tiled 都让一个 CUDA thread 负责一个目标粒子 `i`。线程独立遍历全部源粒子 `j`，计算自己的 `ax/ay/az`，所以总复杂度是 `O(N²)`。

Euler 每步启动：

1. acceleration kernel；
2. Euler position/velocity update kernel。

Leapfrog 在循环前先启动一次 acceleration kernel，之后每步启动：

1. kick-drift kernel；
2. acceleration kernel；
3. second-kick kernel。

这个顺序保持了 `a(t) → 半步速度 → 整步位置 → a(t+dt) → 半步速度`，没有为了少一次 launch 改变积分公式。

### 数据和访存

文件读取使用 AoS `Particle`，原有 GPU 计算已经是 SoA：`x/y/z/vx/vy/vz/mass` 分离数组。baseline 每个线程先读自己的三个坐标，内层循环逻辑上再读约 `4×(N-1)` 个 float，最后写一次 `ax/ay/az`。同一个 warp 在某次循环里会读取相同的 `j`，所以现代 GPU 的缓存和广播会减少真实 DRAM transaction；`4N` 是代码层面的 load 数量，不等于实际 DRAM 流量。

原有 tiled 已经按 block 把 `x/y/z/mass` 搬到 shared memory，尾 tile 也按实际数量处理。一个 tile 的源粒子从 global memory 加载一次，随后由 block 内线程重复使用。每个 block 对全部源粒子读取约 `4N` 个 float，所有 block 合计约 `4N×ceil(N/block_size)`，相比 baseline 的约 `4N²` 次逻辑读取减少约一个 block-size 的复用倍数。

两个 kernel 原本也已经做到：

- `softening²` 正确加入 `dist2`；
- 使用 `rsqrtf` 和三次乘法得到 `1/r³`；
- `ax/ay/az` 放在寄存器局部变量中累加；
- 完整遍历以后才把三个加速度写回 global memory；
- 轨迹只在记录步 D2H，结束时再拷一次完整状态做正确性验证。

### 修改前最明显的问题

每个 timestep 都会 `cudaEventSynchronize`。这不会把磁盘写入算进 CUDA Event，但会强迫 CPU 每步等待 GPU，破坏连续 kernel 提交，也让小规模 wall time 受到大量同步影响。另一个缺口是测试只比较最终位置，没有比较最终速度和三个 GPU 版本之间的轨迹。

## 2. 本轮修改

- 保留 `compute_acceleration_baseline` 和原来的四 shared-array `compute_acceleration_tiled`。
- 新增 `compute_acceleration_tiled_float4`：global/shared 中均用 `float4(x,y,z,mass)`，并加入 `__restrict__` 和四次内层循环展开提示。
- 新增 `--kernel tiled-float4` 和 `--block-size N`，可以独立比较算法与 block size。
- CUDA Event 改成按轨迹记录区间分段计时；记录点之间连续提交 kernel，只在 D2H 前同步。
- 增加最终速度最大/RMS 误差、三种轨迹互比、两体轨道半径检查。
- 增加 `NBODY_FAST_MATH` 和 `NBODY_PTXAS_INFO` CMake 选项，均默认关闭。
- benchmark 支持多 kernel、多 block size、多次重复，并输出原始 CSV 与中位数汇总。

输入文本、配置格式和 `[record][particle][xyz]` 二进制轨迹格式均未改变。

## 3. Shared memory 与寄存器

两个 tiled 版本每个 block 的动态 shared memory 都是：

```text
block_size × 4 × sizeof(float) = block_size × 16 bytes
```

对应 64/128/256/512 threads 时为 1/2/4/8 KiB。`tiled-float4` 没有减少 shared-memory 字节数，它把四次标量 global load 组织成一次 16-byte 对齐向量 load，主要减少加载指令并让布局更明确。

sm_86 的 ptxas 实测：

| acceleration kernel | registers/thread | spill |
|---|---:|---:|
| baseline | 40 | 0 |
| tiled | 28 | 0 |
| tiled-float4 | 25 | 0 |

这次 float4 路径没有增加 register pressure，也没有 local-memory spill。不同 CUDA 版本和架构可能得到不同结果，可用 `-DNBODY_PTXAS_INFO=ON` 重新检查。

## 4. 正确性与精度

CTest 会验证 baseline、tiled、tiled-float4、Euler、Leapfrog 和两体轨道。默认构建的三种 GPU 路径通过轨迹互比；CPU 对照同时输出：

- `max/rms_position_error`；
- `max/rms_velocity_error`；
- momentum error；
- relative energy error。

`float4` 只改变布局，`__restrict__` 只提供无别名信息，默认没有启用 fast math。内层展开可能改变编译器的指令调度，但没有改变 `j` 的累加顺序。`NBODY_FAST_MATH=ON` 会改变除法、平方根和部分浮点规则，因此作为单独实验构建；本机 fast-math 两体测试的最大位置误差为 `2.1275e-6`、最大速度误差为 `2.9115e-6`、相对能量误差为 `1.17525e-5`，仍通过当前阈值，但其他数据集必须重新验证。

## 5. 实测结果

### N=4096，200 步

| kernel | 最佳 block size | GPU simulation 中位数 |
|---|---:|---:|
| baseline | 64 | 59.236 ms |
| tiled | 64 | 66.153 ms |
| tiled-float4 | 128 | 64.926 ms |

同为 block size 128 时，tiled-float4 比普通 tiled 快约 7.3%；最佳 baseline 仍比最佳 tiled-float4 快约 9.6%。

### N=16384，50 步

| kernel | 最佳 block size | GPU simulation 中位数 |
|---|---:|---:|
| baseline | 64 | 60.528 ms |
| tiled | 256 | 68.608 ms |
| tiled-float4 | 256 | 67.760 ms |

同为 block size 256 时，tiled-float4 比普通 tiled 快约 1.2%；baseline 仍然最快。完整三次原始数据位于 `results/optimization_benchmark*/benchmark.csv`。

Nsight Compute 对 `N=4096` 的单次 acceleration kernel 显示网格偏小：64-thread baseline 只有 64 blocks，256-thread tiled 只有 16 blocks，而 RTX 3080 有 68 个 SM。DRAM throughput 很低，说明这组输入并非简单的 DRAM 带宽瓶颈。baseline 的同址读取被缓存/广播有效处理，tiled 的两次同步和 shared-memory 指令反而成为额外成本。因此当前结果没有把“global load 逻辑数量下降”误写成“tiled 必然更快”。

## 6. 运行方法

正确性：

```bash
ctest --test-dir build -C Release --output-on-failure
```

单次对比：

```bash
./build/nbody particles.txt config.txt --kernel baseline --block-size 128
./build/nbody particles.txt config.txt --kernel tiled --block-size 128
./build/nbody particles.txt config.txt --kernel tiled-float4 --block-size 128
```

批量 benchmark：

```bash
python scripts/benchmark.py ./build/nbody --sizes 4096 16384 \
  --steps 100 --block-sizes 64 128 256 512 --repeats 3
```

ncu 的具体命令和推荐指标见 README 的 “Nsight Compute / Nsight Systems” 小节。Profiler 会重放 kernel，不能把 profiler 运行时生成的 performance.log 当作正常性能结果。

## 7. 下一轮可继续做的事

后续最值得验证的是：提高每次 launch 的并行 block 数、尝试一个 thread 处理多个目标粒子时的寄存器复用、用 CUDA Graph 降低大量小 kernel 的 launch 开销，以及只在单独实验分支测试 fast math。它们都应继续保持 direct `O(N²)`、现有积分顺序和文件格式。
