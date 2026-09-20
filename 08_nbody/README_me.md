# GPU 加速 N 体引力模拟器 —— 最终提交版

> InfiniTensor 2026 夏季训练营 · CUDA 方向 · 选题 8「N 体引力模拟与可视化」

用 CUDA 并行计算数千到数十万个天体之间的引力相互作用，输出轨迹数据供
Python 可视化，并提供展示**星团演化**、**轨道扰动**、**局部碰撞风险**
三种现象的动画脚本。

**本目录是最终提交版。** 力计算采用优化阶梯的终点档 **V3**
（shared memory tiling + `float4` 打包 + `rsqrtf`），它是 4090D 上
192 组合扫参里综合最优的一档：

- 在 6 个测试规模（4096 → 262144）中的 4 个上最快、2 个上持平
- 代码量不到次优档 V4 的一半（无模板实例化、无跨步索引算术）
- 实测能量误差 1.3e-5 ~ 1.8e-5，与全部其他档位一致

逐档优化的对比数据与历次实测日志见 **`n_bodyversion.md`**（最终提交报告，自包含：
实现思路、数值积分、优化方法、正确性验证、性能分析、nsys、未来提升、逐文件索引）。
完整五档阶梯（V0..V4 + coarsen 扫参）的源码在开发版 `n_version5/` 中。

---

## 目录

- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [构建](#构建)
- [快速开始](#快速开始)
- [输入输出格式](#输入输出格式)
- [命令行参考](#命令行参考)
- [正确性验证](#正确性验证)
- [性能结果](#性能结果)
- [Profiling](#profiling)
- [可视化](#可视化)
- [已知局限](#已知局限)
- [文档索引](#文档索引)

---

## 目录结构

```
final_version/
├─ CMakeLists.txt          # 双架构构建（sm_61 本地 / sm_89 远程），MSVC 与 gcc 双平台
├─ README.md               # 本文件（构建/运行/CLI/格式/验证/性能速查）
├─ n_bodyversion.md        # **最终提交报告**（自包含：实现思路/数值积分/优化方法/验证/性能/nsys/未来提升/逐文件索引）
├─ include/
│  ├─ nbody_types.h        # SoA 布局、real4 打包、精度开关、FLOP 计数约定
│  ├─ kernels.cuh          # 力计算 kernel 声明 + V3 各层优化的设计理由
│  ├─ cuda_utils.cuh       # CUDA_CHECK、事件计时、设备信息
│  ├─ sim_params.h         # 参数文件解析
│  ├─ trajectory_io.h      # 粒子初值解析 + 轨迹写出（二进制/CSV）
│  ├─ diagnostics.h        # 能量/动量/角动量（主机侧参考实现，double 累加）
│  ├─ diagnostics_gpu.cuh  # GPU 诊断量（默认路径）
│  ├─ snapshot_stream.cuh  # pinned 双缓冲 + 独立流的轨迹传输管线
│  ├─ cpu_reference.h      # 三档 CPU 基线
│  └─ cpu_kernel_inl.h     # 三档 CPU 基线共享的标量内核
├─ src/
│  ├─ main.cu              # 模拟主程序
│  ├─ kernels.cu           # 力计算 kernel + 积分器 kernel + 设备内存管理
│  ├─ ic_generator.cpp     # 初值生成器 nbody_ic（8 种预设）
│  ├─ sim_params.cpp       # 参数文件解析
│  ├─ trajectory_io.cpp    # 粒子文件解析 + 轨迹写出
│  ├─ diagnostics.cpp      # 主机侧诊断量 + 二体轨道要素
│  ├─ diagnostics_gpu.cu   # GPU 诊断量 kernel
│  ├─ snapshot_stream.cu   # 轨迹传输管线（async / sync 两条路径）
│  ├─ cuda_utils.cu        # 设备查询与计时实现
│  ├─ cpu_reference.cpp    # 三档 CPU 基线（含 BenchmarkCpuForceEval）
│  └─ cpu_naive.cpp        # 第 1 档 CPU 基线（独立编译单元，禁用向量化）
├─ tools/
│  ├─ visualize.py         # FuncAnimation 2D/3D 动画
│  ├─ plot_energy.py       # 守恒量漂移曲线（从运行日志解析，Euler vs Leapfrog）
│  ├─ plot_scaling.py      # 性能扫参曲线 + 加速比柱状图（读 bench 的产物）
│  └─ check_collision_energy.py  # 并合过程的能量分解分析
├─ data/                   # 参数文件（初值文件由 nbody_ic 生成）
├─ video/                  # **成品动画与图表**（交付物本体）
│  ├─ cluster.mp4          # 星团演化（Plummer N=65536，3D）
│  ├─ two_body.mp4         # 轨道扰动（二体圆轨道 N=2，2D）
│  ├─ two_body_euler.mp4   # Euler 对照（非辛积分器的轨道漂移）
│  ├─ collision.mp4        # 局部碰撞风险（双球对撞并合，3D）
│  ├─ energy_comparison.png # Leapfrog vs Euler 能量漂移对比
│  ├─ energy.png           # 能量分项随时间变化
│  └─ scaling.png          # 性能曲线三面板
└─ scripts/                # 远程一键脚本
   ├─ build.sh             # 配置 + 编译
   ├─ validate.sh          # 正确性门禁（PASS/FAIL 表）
   ├─ bench.sh             # 性能扫参
   ├─ profile.sh           # ncu / nsys 分析
   └─ remote_all.sh        # 顶层入口，一条命令跑全流程
```

树中**没有 `results/`**：那里放的是运行日志与中间数据（`validate/`、`bench/`、
`profile/` 三个子目录 + 顶层 `remote_all_*.log`），由 `scripts/` 现场创建，
**不随代码提交**（详见下文「结果日志」）。提交的成品是 `video/` 下的 7 个文件
和它们的生成脚本。

---

## 环境要求

| 组件 | 版本 |
|---|---|
| CUDA Toolkit | ≥ 11.0（实测 12.6 / 12.8） |
| CMake | ≥ 3.18 |
| 宿主编译器 | gcc ≥ 9 或 MSVC ≥ 2019 |
| GPU | 计算能力 ≥ 6.1（实测 MX330 sm_61 / RTX 4090D sm_89） |
| Python（可视化） | ≥ 3.8，`numpy`、`matplotlib` |

MSVC 需要版本 ≥ 2019（19.20）才能正确处理 C++17 的 CUDA 分离编译。
源码注释为 UTF-8，MSVC 下必须加 `/utf-8`（`CMakeLists.txt` 已处理）。

---

## 构建

### Linux

```bash
cd final_version
bash scripts/build.sh              # 默认 sm_89，产出 build/

# 或手动指定架构
NBODY_ARCH=61 bash scripts/build.sh
```

`build.sh` 会检测 `build/CMakeCache.txt` 里的源码路径是否与当前目录一致；
不一致（例如从别的机器拷过来的 `build/`）会自动重建，避免 CMake 拒绝复用缓存。

> **架构要按实际显卡填。** `NBODY_ARCH=89` 是默认值（RTX 4090D）。CMake 会同时
> 生成 PTX，所以在更新的架构（如 RTX 5090 / sm_120）上也能直接跑——但那是
> **驱动 JIT** 出来的 SASS，未必最优。要拿原生 SASS 的性能数字请显式指定：
> `NBODY_ARCH=120 bash scripts/build.sh`（sm_61 = MX330，sm_89 = 4090D）。
>
> 若报 `No CMAKE_CUDA_COMPILER could be found`，是 `nvcc` 不在 PATH：
> `export PATH=/usr/local/cuda/bin:$PATH`，或直接
> `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`。

### Windows

`cmake` 不在 PATH 时（VS 自带的那份）：

```bat
set CMAKE="D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
%CMAKE% -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61
%CMAKE% --build build --config Release -j
```

### 编译开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES` | `61;89` | 目标架构。同时编译两档保证本地能抓到 sm_89 的编译期错误 |
| `-DNBODY_FP64=ON` | OFF | 力计算切 FP64。**仅用于小 N 验证**，用来区分算法误差与浮点误差；4090D 的 FP64 吞吐只有 FP32 的 1/64，开它是自废武功 |
| `-DNBODY_FAST_MATH=ON` | OFF | 给 `.cu` 加 `-use_fast_math`。用来量化 fast-math 的收益与精度代价，不常开 |

---

## 快速开始

三种现象的完整流程（以星团演化为例）：

```bash
B=./build

# 1) 生成初值（Plummer 球，N=65536）
$B/nbody_ic --preset plummer -n 65536 --out data/plummer_65536.txt

# 2) 模拟
$B/nbody_sim --bodies data/plummer_65536.txt \
             --params data/params_cluster.txt \
             --block 128 \
             --out results/plummer.bin \
             --log results/plummer.json

# 3) 可视化（2D 动画；加 --dim 3 出 3D，加 --save out.mp4 存文件）
python3 tools/visualize.py results/plummer.bin
```

### 三种现象对应命令

```bash
# ① 星团演化：Plummer 球在自身引力下演化，位力比围绕 1 振荡
$B/nbody_ic --preset plummer -n 65536 --out data/plummer.txt
$B/nbody_sim --bodies data/plummer.txt --params data/params_cluster.txt \
             --out results/cluster.bin

# ② 轨道扰动：旋转盘 + 中心核球，出现螺旋密度波
$B/nbody_ic --preset disk -n 32768 --out data/disk.txt
$B/nbody_sim --bodies data/disk.txt --params data/params_bench.txt \
             --out results/disk.bin

# ③ 局部碰撞风险：两个 Plummer 球在束缚轨道上并合
$B/nbody_ic --preset collision -n 4096 --sep 8 --vfrac 0.7 --impact-param 1.5 \
            --out data/collision.txt
$B/nbody_sim --bodies data/collision.txt --params data/params_collision.txt \
             --out results/collision.bin --diag-interval 5000
python3 tools/check_collision_energy.py results/collision.bin --params data/params_collision.txt
```

并合过程的动力学判据（位力比 2T/|W|）会随 `--diag-interval` 打印：

```
step       0  2T/|W|=0.9830    两球各自平衡
step    7500  2T/|W|=1.4762    第一次近星点，动能激增
step   30000  2T/|W|=0.9917    穿过后回落
step  150000  2T/|W|=1.0043    重新趋于平衡
```

### 一键全流程（云上）

```bash
bash scripts/remote_all.sh          # 构建 + 验证 + 扫参 + profiling，输出可贴回摘要
bash scripts/remote_all.sh quick    # 只构建 + 验证（省机时）
```

---

## 输入输出格式

### 输入 1：粒子初值文件（文本）

每行一个粒子，7 个字段：

```
x y z vx vy vz mass
```

示例：

```
0.0 0.0 0.0  0.0  0.0  0.0  1.0e6
1.0 0.0 0.0  0.0  1.0  0.0  1.0
0.0 1.0 0.0 -1.0  0.0  0.0  1.0
```

> ⚠️ **作业示例数据在物理上是错的，不要直接用它做二体验证。**
> 示例里 `mass=1e6` 的中心天体 + 半径 1 处速度 1.0：圆轨道速度应为
> $\sqrt{GM/r} = 1000$，给 1.0 意味着粒子近似径向自由落体砸向中心，
> 形不成轨道。本项目的二体验证初值由 `nbody_ic` 自己构造：
> 等质量双星各 $m=1$、间距 $d=1$、各自速度 $\sqrt{2}/2 \approx 0.7071$、
> 周期 $T = 2\pi/\sqrt{2} \approx 4.442883$。

### 输入 2：模拟参数文件（`key = value`）

```
dt = 1e-3               # 时间步长
num_steps = 1000        # 总步数
record_interval = 100   # 每多少步记录一次位置
G = 1.0                 # 引力常数（归一化单位）
softening = 1e-4        # 软化因子，必须 > 0
integrator = "leapfrog" # euler / leapfrog
format = "binary"       # binary / csv
```

支持 `#` 与 `//` 注释、等号两侧任意空白、value 上的引号。未出现的键保留
默认值；遇到无法识别的键会**报错而非静默忽略**——静默忽略拼错的键会导致
"我明明设了 dt 却没生效"这类难查的问题。

`softening` 必须 > 0：kernel 刻意不做 `j == i` 的分支跳过（`r2 = eps²` 使
自交互贡献恰好为 0），`softening = 0` 会产生 `0/0 = NaN` 并静默污染全部结果，
程序在启动时校验。

**软化因子要按初值调**：二体验证用 `eps=1e-4`（粒子不会靠近），
星团与对撞用 `1e-2 ~ 5e-2`，否则近距离两体会数值爆炸。

### 输出 1：轨迹数据文件（二进制，粒子主序）

需求指定的格式：

```
P (int32)          # 粒子数
R (int32)          # 记录点数
然后 for p in 0..P-1:
        for r in 0..R-1:
            x, y, z    # 3 个 float32
```

> ⚠️ **这是本项目最容易出错的地方。** GPU 天然产出**记录主序**（每隔
> `record_interval` 步一个全体快照），写文件时必须转置成**粒子主序**。
> 写反了文件大小、P、R 全部对得上，程序不报任何错，只有动画会完全错乱。
> 所以 `validate.sh` 段 1 专设一项**字节级交叉判定**：先让 `num_steps=1` +
> 超大 `record_interval` 只落 1 帧（此时两种展平等价，是初值的权威字节布局），
> 再跑多帧版本，断言后者“每个粒子的第 0 帧”拼起来与前者的载荷逐字节相等。

Python 侧读取：

```python
import numpy as np
with open(path, "rb") as f:
    P, R = np.fromfile(f, dtype=np.int32, count=2)
    data = np.fromfile(f, dtype=np.float32).reshape(P, R, 3)   # 粒子主序
    # 若想按记录取全体快照： data[:, r, :] 是一个时刻的全部位置
```

`R` 包含 `t=0` 这个记录点，即 `R = num_steps // record_interval + 1`。

`format = "csv"` 时输出 `particle_id, step, x, y, z` 行。**CSV 明显更低效**：
同样内容体积约为二进制的 6–8 倍，且解析慢得多。本版保留它仅为在报告中
量化这个差距。

### 输出 2：性能日志

stdout 打印，`--log` 可另存 JSON：

| 字段 | 含义 |
|---|---|
| `total_ms` / `ms_per_step` | 总耗时 / 每步平均耗时 |
| `particle_steps_per_sec` | 粒子·步/秒 |
| `gflops` | 吞吐（按 20 flop/interaction 折算） |
| `device_mem_used_mib` | 显存占用 |
| `rel_energy_error` | 首末相对能量误差 |
| `rel_angular_momentum_error` | 首末相对角动量漂移（有心力下严格守恒量） |
| `angular_momentum_initial` / `_final` | 首末角动量模长 |
| `momentum_initial` / `_final` | 首末总动量模长（只在舍入级别守恒，见 §4.3） |
| `cpu_force_eval_ms` | 三档 CPU 基线各一次力评估的耗时 |
| `speedup_vs_cpu_openmp` | 对 OpenMP 档的加速比 |

> **FLOP 计数约定**：每次两体交互 ≈ **20 flop**（3 减 + 5 求 r² + 1 加 eps²
> + 1 rsqrt + 2 乘求 inv³ + 1 乘 m_j + 6 累加 = 19，取整 20，
> 与 GPU Gems 3 ch.31 一致）。报告里所有 GFLOP/s 均按此约定，
> 否则数字无法与文献比较。

---

## 命令行参考

### `nbody_sim`

```
nbody_sim --bodies <file> --params <file> [options]

必需：
  --bodies <file>         粒子初值文件
  --params <file>         模拟参数文件

输出：
  --out <file>            轨迹输出路径（默认 trajectory.bin）
  --log <file>            另存性能日志为 JSON

运行：
  --block <int>           CUDA block size（默认 128，4090D 上 V3 的最优值）
  --diag-interval <int>   每 N 步输出诊断量（O(N²)，大 N 下要放大）
  --device <int>          CUDA 设备号

验证：
  --cpu-check             GPU vs CPU 加速度逐元素比对
  --cpu-bench             跑三档 CPU 基线以得出加速比
  --check-diag            GPU 诊断量 vs 主机侧参考实现逐项比对
  --cpu-diag              诊断量改走主机侧路径（很慢，仅供对照）
  --sync-copy             轨迹记录走同步 pageable 拷贝（A/B 对照用）
```

### `nbody_ic`（初值生成器）

```
nbody_ic --preset <name> -n <count> --out <file> [options]

  --preset <name>  two_body | two_body_ecc | plummer | king
                   | disk | uniform | collision | solar_system
  -n, --count <int>        粒子数
  --out <file>             输出路径
  --seed <int>             随机种子（默认 20260821）
  --w0 <float>            King 模型的中心势
  --imf / --imf-max       Plummer/King 的初始质量函数
  --sep <float>            碰撞预设：两球初始间距
  --vfrac <float>          碰撞预设：相对速度 / 逃逸速度（0.7 = 束缚轨道）
  --impact-param <float>   碰撞预设：碰撞参数 b / 半长轴（1.5 = 螺旋并合）
```

> 碰撞预设的 `-n` 是**粒子总数**（在两球之间平分），不是每个球的粒子数。

---

## 正确性验证

```bash
bash scripts/validate.sh
```

8 段共 16 项检验，任一项 FAIL 则退出码非零。**RTX 5090（sm_120）上 16/16 全 PASS**（2026-09-19 实测，`results/remote_all_20260919_104218.log`）：

| 检验项 | 判据 | 实测 |
|---|---|---|
| 二进制粒子主序布局 | 字节级交叉判定（段 1） | PASS |
| 二体圆轨道运行（10 周期） | 退出码 | PASS |
| Leapfrog 能量误差 | < 1e-3 | `7.274×10⁻⁷` |
| 半长轴漂移 | < 1% | `a = 1.000001` |
| 偏心率保持圆 | e < 0.01 | `e = 0.000015` |
| **轨道周期偏差** | < 1%（解析值 4.442883） | `T = 4.442888`（偏差 **1.1×10⁻⁶**） |
| **GPU vs CPU 加速度（N=2）** | 相对误差 < 2e-3 | **0.000e+00**（位级重合） |
| 偏心轨道运行 | 退出码 | PASS |
| 偏心轨道偏心率 | 匹配解析值 0.36（容差 0.02） | `e = 0.359979` |
| **Leapfrog 优于 Euler** | 误差比 | **182568×** |
| Plummer 多体运行 | 退出码 | PASS |
| 多体能量误差 | < 1e-2 | `9.258×10⁻⁸` |
| **多体角动量漂移** | < 1e-2 | **8.813×10⁻⁷**（两体 3.636×10⁻⁷） |
| GPU vs CPU 加速度（N=4096） | 逐元素比对 | PASS（6.938×10⁻⁷） |
| GPU 诊断量 vs 主机参考 | 逐项 < 1e-10 | ≤ `2.4×10⁻¹⁵` |
| CSV 输出 | 退出码 | PASS |

### 最有说服力的两组数据

**Euler vs Leapfrog**：同一初值、同一 dt、同一步数，Euler 的
`|ΔE/E0| = 1.328×10⁻¹`，Leapfrog 只有 `7.274×10⁻⁷`，相差 **182568 倍**。
Euler 跑完后半长轴从 1.0 涨到 1.153（螺旋外扩 15%），直接印证
"一阶非辛方法能量单调漂移"的理论预期。

> 这个对照在两代卡上都做过，结果本身很有信息量：Euler 的 `1.328×10⁻¹` 在
> 4090D 与 5090 上**逐位相同**（由 $O(dt)$ 截断误差主导，与架构无关）；
> Leapfrog 则从 `3.743×10⁻⁶`（sm_89）降到 `7.274×10⁻⁷`（sm_120）——
> 它的残差由 `rsqrtf` 的 SFU 近似精度主导，而 sm_120 的近似更准。
> 于是"Leapfrog 优于 Euler"的倍数从 35480× 变成 **182568×**。

**GPU vs CPU 逐元素比对得到 0.000e+00 而非某个小误差**：N=2 时两边算法
完全一致——GPU 一个线程和 CPU（OpenMP）一个线程都是对同一个目标粒子按
`j = 0..N-1` 顺序完整累加，累加顺序相同，`sqrtf`/`std::sqrt` 都是 IEEE
正确舍入。N=4096 时因为 `rsqrtf` 的近似而出现约 1e-6 的相对误差。
**位级重合是好结果不是异常**，说明两侧实现的数学表达式确实一致。

### 动量守恒（诚实汇报）

Plummer N=4096 跑 2000 步，`|p|` 从 `9.646e-11` 涨到 `2.967e-08`——
**只在浮点舍入级别守恒，不是精确守恒**。原因是力计算 kernel 对每个粒子
独立求和，没有利用 $F_{ij} = -F_{ji}$ 的对称性把一对相互作用只算一次。
误差来源是 FP32 累加的随机游走，量级约 $\sqrt{\text{步数} \cdot N}$。
**这不是 bug**，程序在输出里显式说明。

---

## 性能结果

全部数字来自 **NVIDIA RTX 4090 D**（sm_89, Ada, 24564 MiB, driver
570.124.06；CPU: Intel Xeon Platinum 8358P @ 2.60GHz, 128 核；
CUDA 12.8, gcc 13.3.0, Ubuntu 24.04）。

> RTX 5090（sm_120）的跨代复测数据在本节末尾，完整分析见
> `n_bodyversion.md` §3.9。

> **本地 MX330（sm_61, Pascal）的数字不在此表中**，它只用于正确性验证与
> 相对趋势。两个平台的数字不混用、不换算。

### V3 吞吐（uniform 球，1000 步，softening=0.01）

| N | best block | ms/step | GFLOP/s | particle-steps/s | 显存 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 128 | 0.0825 | 4065 | 4.96×10⁷ | 356 MiB |
| 16384 | 256 | 0.3917 | 13708 | 4.18×10⁷ | 356 MiB |
| 32768 | 128 | 0.9909 | 21673 | 3.31×10⁷ | 356 MiB |
| 65536 | 128 | 3.2272 | 26617 | 2.03×10⁷ | 358 MiB |
| 131072 | 128 | 11.3954 | 30152 | 1.15×10⁷ | 362 MiB |
| **262144** | **256** | **43.7845** | **31390** | 5.99×10⁶ | 368 MiB |

显存占用恒定在约 356–368 MiB（pos/vel/acc/mass 的 SoA + `real4` 打包缓冲），
N=262144 时设备数组本身也仅约 12.6 MB——**瓶颈是轨迹文件体积，不是显存**。

### CPU 加速比（N=65536，128 OpenMP 线程，同轮配对）

| 档位 | 单次力评估 | 对 V3 的加速比 |
|---|---:|---:|
| `cpu_naive`（单线程标量） | 18664 ms | 5784×（稻草人，仅供参考） |
| `cpu_scalar_o3`（单线程 + 向量化） | 7997 ms | 2478× |
| **`cpu_openmp`（128 线程 + 向量化）** | **3628 ms** | **1124×** ← 报告引用 |

> **加速比必须与同轮的 CPU 数字配对。** 云容器的 CPU 负载逐轮波动
> （历次实测的 `cpu_openmp` 分别为 2970 / 4401 / 3662 / 3628 / 3920 / **4244** ms；
> 最后一个 4244 ms 是 5090 轮的 384 线程值，对应 `scripts/bench.sh` 写出的
> `results/bench/speedup_n65536.log`（加速比 1773×）），
> 跨轮比较加速比没有意义。上表是同一轮实测，`3628 ms ÷ 3.2272 ms ≈ 1124×`。
> 报告引用的分母一律是 `cpu_openmp`；`cpu_naive` 的加速比是稻草人对比，
> 不作主结论。

### 与硬件极限的对照

| 维度 | 4090D | 实测 | 占比 |
|---|---:|---:|---:|
| FP32 峰值吞吐 | 73.5 TFLOP/s | 31.4 TFLOP/s（v5 轮；历史峰值 32.6） | **43%（峰值 44%）** |
| DRAM 带宽上限（按 AI≈1.25 flop/byte 折算） | ~1.26 TFLOP/s | 31.4 TFLOP/s | **24.9×** |
| 显存容量 | 24 GB | 0.37 GB | 1.5% |

> FP32 峰值 = 14592 core × 2 flop/cycle × ~2.52 GHz = **73.5 TFLOP/s**。
> 注意别把非 D 版 RTX 4090 的 82.6 TFLOP/s（16384 core）当成这块卡的峰值。

**为什么只有 ~44% 的 FP32 峰值**：内层循环每次交互必含 1 次 rsqrt（SFU，
吞吐低于 FMA）与 1 次除法（DP 单元，吞吐约为 FMA 的 1/4），外加 smem 加载
与循环控制的整数运算、8 路归约累加的依赖链约束 ILP。瓶颈在 SFU/DP 吞吐
与 ILP，不在 FMA。GPU Gems 3 ch.31 报告的同类实现峰值约 50% 理论吞吐，
本项目的 44% 属于同一量级的合理水平。

**为什么实测吞吐超出"DRAM 带宽上限"24.9 倍**：4090D 有 72 MB L2，而
N=262144 时源数组总共只有约 4.2 MB，**整个数据集常驻 L2**；加上同一 warp
的 32 个线程在同一时刻读同一个源粒子（广播访问），实际吃到的是接近 L2
带宽而非 DRAM 带宽。**这个发现直接下调了对访存优化的预期**——tiling 的传统
卖点是"逃离 DRAM 带宽上限"，但这里 DRAM 瓶颈本来就不存在。

### 优化空间（诚实汇报）

V3 相对朴素实现的加速比是 2.3×（N=65536：26617 vs 11532 GFLOP/s），
其中绝大部分来自 `rsqrtf`（约 2.0×），访存优化（tiling + `float4`）
只贡献了约 16%。这与教科书对 tiling 的预期（数倍提升）相差很远，
原因就是上一条：本项目的工作负载从一开始就不是 DRAM-bound。

### RTX 5090 复测（2026-09-19，sm_120）

同一份代码搬上 RTX 5090 重跑，作为跨代验证（完整日志
`results/remote_all_20260919_104218.log`；RTX 5090 sm_120, 170 SM,
31.4 GiB, driver 610.43.02；Xeon 8358P, **384** 核）：

| N | best block | ms/step | GFLOP/s |
|---:|---:|---:|---:|
| 4096 | 128 | 0.0675 | 4973 |
| 16384 | 128 | 0.2590 | 20726 |
| 32768 | 256 | 0.6732 | 31901 |
| 65536 | **512** | 2.1489 | 39973 |
| 131072 | 128 | 8.1219 | 42305 |
| **262144** | **128** | **30.4213** | **45178.6** |

> 本表出自运行产物 `results/remote_all_20260919_104218.log`。同卡另一次扫参
> （`results/bench/results.csv`）为 30.768 ms / 44669.4 GFLOP/s，差 **1.1%**。
> 两份都不随代码提交，引用时指哪份用哪份。详见 `n_bodyversion.md` §3.9。

| 指标 | 4090D | **5090** | 说明 |
|---|---:|---:|---|
| FP32 峰值 | 73.5 TFLOP/s | **106.6 TFLOP/s** | 170 SM × 128 × 2 × 2.45 GHz |
| 实测峰值吞吐 | 32591 GFLOP/s | **45178.6 GFLOP/s** | **1.39×** |
| FP32 峰值利用率 | 44.4% | **42.4%** | 几乎不变 |
| 相对 SM 数的扩展效率 | — | **93%** | 170/114 = 1.49× |
| N=65536 单步（b128） | 3.227 ms | 2.370 ms | — |
| CPU 加速比 | 1095× | **1397.8×** | 分母从 128 线程变 384 线程 |

**利用率几乎不动（44.4% → 42.4%）是这一轮最重要的结果**：SM 数 +49%、
roofline 脊点从 72.9 降到 59.5 flop/byte（带宽相对更充裕），利用率却没有
随两代卡“带宽/算力比”的变化而漂移 → 瓶颈是**每 SM 的指令吞吐**
（SFU rsqrt + 除法 + ILP），不是访存。

**两个必须随数字一起引用的条件**：

1. 本轮构建用的是默认 `NBODY_ARCH=89`，在 sm_120 上跑的是 **PTX JIT**
   出来的 SASS，未必最优。要拿原生 sm_120 的数字需重跑
   `NBODY_ARCH=120 bash scripts/remote_all.sh`（**未测量**）。
2. nsys 的占比依赖 profile harness 的口径：20 步 + 2 次 $O(N^2)$ **FP64**
   诊断时，诊断 kernel 占 GPU 时间 **89.6%**，力计算只占 10.3%。
   “力计算占 100%”只在**长步数 + 正常 `--diag-interval`** 下成立
   （详见 `n_bodyversion.md` §3.10 与附录 F）。

另一个可复现但尚未解释的观测：5090 上最优 block 不再恒定——32768→256、
**65536→512（比 128 快 10.3%）**、131072/262144→128，而 4090D 上 128 基本通吃。
10.3% 远超轮间方差（<0.3%），但成因需要 ncu 才能定论。

---

## Profiling

```bash
bash scripts/profile.sh
```

产出 `results/profile/` 下的 `ncu_summary.txt` 与 `nsys_stats.txt`。

### nsys：力计算是唯一热点

N=65536 跑 20 步的 GPU kernel 时间分布：

```
Time(%)  Total(ns)   Instances  Kernel
 100.0   187665139       21     KernelComputeAccV3   ← 力计算
   0.0       41248       20     KernelKickDrift
   0.0       31648       20     KernelKick
   0.0        3456        2     KernelPackPositions
```

力计算占 GPU 时间的 **100%**。拷贝方面，20 步内 D2H 71.8% + H2D 28.2%
总计约 0.3 ms，量级远小于计算。这确认了"力计算是唯一值得优化的热点"，
并支撑了"轨迹传输管线对当前 record_interval 收益有限"的判断。

### ncu：云平台权限限制（已知）

`ncu --set full` 在云 GPU 租赁平台上未能采集到 Speed-of-Light /
Occupancy / 寄存器指标，原因是**驱动权限限制**（`ERR_NVGPUCTRPERM`）：
即便容器内是 root，性能计数器访问仍由宿主机驱动策略控制
（需 `NVreg_RestrictProfilingToAdminUsers=0`），租户容器内无法自行解决。

应对：
- SOL / roofline 图无法用 ncu 直接产出，改用 nsys 的 kernel 时间数据 +
  理论 roofline 公式手算（见上面的吞吐表与 L2 缓存分析）。
- 报告如实说明这一限制，这本身也是诚实汇报的一部分。
- `scripts/profile.sh` 的 ncu 部分无需修改，换到有 profiling 权限的环境
  直接重跑即可补全。

---

## 可视化

```bash
# 2D 动画，先看看效果（不存文件时直接弹窗，不需要 ffmpeg）
python3 tools/visualize.py results/plummer.bin

# 3D 动画，存成 mp4（需 ffmpeg；后缀改成 .gif 则用 pillow）
python3 tools/visualize.py results/plummer.bin --dim 3 --save cluster3d.mp4 --fps 30

# 3D + 拖尾（每个粒子画最近 30 个记录点的轨迹）
python3 tools/visualize.py results/plummer.bin --dim 3 --trail 30 --save trail3d.mp4

# 大 N：限制绘制粒子数并放大点（3D 的 mplot3d 很慢）
python3 tools/visualize.py results/plummer.bin --dim 3 --max-particles 1000 \
                                               --size 4 --save big3d.mp4
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `path` | — | 轨迹文件：`.bin`（粒子主序二进制）或 `.csv` |
| `--dim {2,3}` | 2 | 2D / 3D 动画 |
| `--trail N` | 0 | 拖尾长度（单位：记录点数）；0 = 只画当前点。**代价 ∝ 绘制粒子数**，务必与 `--max-particles` 搭配 |
| `--frames N` | 0 | 只渲染前 N 个记录点（0 = 全部）；先用它出个短预览再上全量 |
| `--max-particles N` | 2000 | 最多绘制多少个粒子（随机抽样；动画帧率的关键旋钮） |
| `--interval MS` | 30 | 弹窗播放时每帧间隔 |
| `--size F` | 2.0 | 点大小 |
| `--save FILE` | — | 存文件而不弹窗：`.gif` 走 pillow，其他后缀（`.mp4`）走 ffmpeg |
| `--fps` / `--dpi` | 30 / 110 | 存文件时的帧率与分辨率 |

云容器上往往没有 ffmpeg：请求 `.mp4` 时脚本会打印 warning 并**自动改存同名
`.gif`**（而不是抛一个与原因无关的 `unknown file extension: .mp4`）。要出 mp4
就装 ffmpeg：`apt-get update && apt-get install -y ffmpeg`。
GIF 无外部依赖（pillow），只是体积大、色彩少；报告用的成品动画用 mp4 更合适。

`visualize.py` 用 `np.fromfile(...).reshape(P, R, 3)` 读二进制轨迹并转成记录主序；
布局写反了**不报错、只会让动画完全错乱**（文件大小、P、R 都对得上），
所以 `validate.sh` 段 1 守着写出侧、`load_trajectory` 的尺寸校验守着读入侧。

**性能提示**：动画开销正比于“帧数 × 绘制粒子数”。N=65536、R=100 的轨迹
渲染一次 3D 动画可能要几分钟，所以默认只抽样 2000 个粒子，并用
`set_offsets` / `_offsets3d` 原地更新而不是重画整幅图。3D 比 2D 慢得多，
建议 `--max-particles 500~1000`。

拖尾的“坑”：早期实现是“一个粒子一个 `ax.plot()`”，4096 粒子 + 200 帧就是
**82 万次** Python 层 artist 更新，表现与卡死无法区分（而且没有任何输出）。
现在所有拖尾装在**一条 `Line3DCollection` / `LineCollection`** 里，每帧只更新
一次 segments，并且存文件时每 10 帧打印一行进度。拖尾仍建议配
`--max-particles 500~1500`。

`tools/check_collision_energy.py` 用于并合现象的能量分解分析（动能 / 势能
分别的时间序列），配合 `--diag-interval` 的位力比输出判断并合是否真实发生。

### 分析图（报告用）

这两个脚本只消费上面那些运行的产物，不重新跑模拟：

```bash
# 守恒量漂移曲线：Euler vs Leapfrog（报告 §4.2 的那张图）
python3 tools/plot_energy.py results/validate/two_body_euler.log \
                            results/validate/two_body_leapfrog.log \
                            --out energy_euler_vs_leapfrog.png

# 性能扫参曲线 + 加速比柱状图（--peak-tflops 必须填本机的 FP32 峰值）
python3 tools/plot_scaling.py --peak-tflops 73.5 --out scaling.png
```

`plot_energy.py` 解析的是 `nbody_sim` 的 stdout，所以那次运行要带
`--diag-interval N`（`validate.sh` 的 §3 与 §5 已经带了 `--diag-interval 1000`，
可直接用 `results/validate/two_body_{euler,leapfrog}.log`）。一张图四个面板：`|dE/E0|`、`|dL/L0|`、`|p|`、`2T/|W|`；
日志里没有 `|L|` 列的旧版本记录也能画（该曲线自动跳过）。终端还会打印
每个日志的终态漂移，可直接抄进报告。

`plot_scaling.py` 读 `results/bench/results.csv`（best block 汇总表同时
打印到终端）与 `results/bench/speedup_n65536.json`（加速比柱状图，柱顶
标绝对倍数）。`--peak-tflops` 是参考线，必须按实际硬件填：4090D 是
**73.5**，MX330 约 1.22；脚本不替你猜是哪块卡。

---

## 已知局限

1. **ncu 指标缺失**：云平台驱动权限限制，SOL / roofline / occupancy /
   寄存器数据未采集，`profile.sh` 无需修改即可在有权限的环境补测。
2. **只保留 V3 一档 kernel**：本版为最终提交做了精简。完整优化阶梯
   （V0..V4 + coarsen 扫参）与逐档对比在开发版 `n_version5/` 中，
   报告 §10–§11 的阶梯数据出自那里，无法在本目录现场复现。
3. **coarsen 中间值未测**：只测过 `{1, 2, 4, 8}`，3、6 等值需要额外的模板
   实例化，未实测。
4. **更大 N 的趋势未验证**：N=262144 是实测上限。24 GB 显存允许到约
   50 万粒子，N>262144 时 V3 是否仍最优未测。
5. **动量非精确守恒**：见上文"动量守恒"一节，这是朴素求和结构的固有性质，
   非 bug。
6. **轨迹文件体积**：N=262144、`record_interval=1` 时单次轨迹约 786 MB。
   程序启动时会打印预估大小，超 1 GiB 时给出警告。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 本文件：构建、运行、格式、CLI、验证、性能 |
| `n_bodyversion.md` | **最终提交报告**：自包含——实现思路、数值积分、优化方法、正确性验证、性能分析、nsys、未来提升、逐文件索引 |

### 复现全部数据

```bash
bash scripts/remote_all.sh          # 构建 + 验证 + 扫参 + profiling
bash scripts/remote_all.sh quick    # 只构建 + 验证
```

数据归档在 `results/` 下（**运行产出，不随代码提交**；`bash scripts/remote_all.sh`
可一键重现）：

- `results/validate/*.log` + `*.json` —— 正确性验证的完整输出
- `results/bench/results.csv` —— 性能扫参（N × block）
- `results/bench/speedup_n65536.json` —— CPU 基线与加速比
- `results/profile/ncu_summary.txt` / `nsys_stats.txt` —— profiling 摘要
- `results/remote_all_*.log` —— 全流程完整日志

> 本仓库提交的成品只有 `video/` 下的动画与图表（以及它们的生成脚本）：
> `.bin` / `.txt` 轨迹动辄上百 MB，且全部可由 `nbody_ic` + `nbody_sim` 重建。
> 文中引用的 `results/...` 路径是 provenance（说明数字出自哪份日志），
> 不是仓库内的文件路径。
