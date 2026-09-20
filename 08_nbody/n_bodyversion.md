# N 体引力模拟与可视化 —— 实现思路、数值积分与优化方法

> InfiniTensor 2026 夏季营选题 8 · CUDA 方向
> 提交路径：`nbody_simulation/<GitHubID>/`
> 提交内容：源码（`src/` `include/`）+ `data/` + `scripts/` + `tools/`
> + `CMakeLists.txt` + `README.md` + 本报告 + `video/` 下 7 个成品动画/图表。
> `build/` 与 `results/` 是运行产物，**不提交**（见 附录 C.9）。

本报告集中阐述三个彼此紧密耦合的核心主题：

1. **实现思路** — 总体架构、数据布局、并行 kernel 设计、轨迹回传与 IO；
2. **数值积分方法** — Leapfrog（KDK）的选型与理由、精度分配、软化因子与时间步长标定；
3. **优化方法** — V0→V4 逐档拆解、roofline 分析、指令级瓶颈推导、coarsen 失效的寄存器解释。

---

## 阅读约定

- **所有性能数字标测量平台。** RTX 5090（sm 120，PTX JIT）、RTX 4090D（sm 89）、MX330（sm 61）。
  三个平台的值**不混用**、**不交叉换算**。
- **未测量的指标写“未测量”**，不用理论值、外推值或 NVIDIA 官方数据代替。
- **FLOP 统一用 20 flop/对相互作用。** 推导见 §3.7。
- “§x.y” 引用均指向本报告自身，除非显式指出外部文档。
- **关于文中引用的 `results/...` 日志路径**：那是一次运行的**产出**，由
  `bash scripts/remote_all.sh` 现场生成，**不随代码提交**。保留路径是为了让
  每个性能数字都指得出它的来源（可对照 `scripts/` 里的采集口径自行重现），
  而不是声称仓库里存着那份文件。随代码提交的只有 `video/` 下的成品动画与图表。

---

## 目录

1. [实现思路](#1-实现思路)
   - 1.1 总体架构与数据流
   - 1.2 数据布局：标量 SoA + 打包 real4
   - 1.3 并行 kernel 设计
   - 1.4 轨迹回传与快照 IO
   - 1.5 诊断量与性能日志
   - 1.6 构建与一键脚本
   - 1.7 输入文件解析与输出文件
   - 1.8 可视化与三种现象演示
2. [数值积分方法](#2-数值积分方法)
   - 2.1 Leapfrog（KDK / velocity‑Verlet）
   - 2.2 Euler 对照
   - 2.3 精度分配：力 FP32、诊断 FP64
   - 2.4 软化因子的标定
   - 2.5 时间步长的收敛性标定
3. [优化方法](#3-优化方法)
   - 3.1 优化阶梯总览
   - 3.2 V0→V1：共享内存 tiling（+12%）
   - 3.3 V1→V2：打包 real4（+9%）
   - 3.4 V2→V3：rsqrtf 替代 1/sqrt（+88%）
   - 3.5 V4：coarsen（几乎无收益，+1.6%）
   - 3.6 为什么不是访存瓶颈：V0 事实上不是 DRAM‑bound
   - 3.7 Roofline 与指令级瓶颈
   - 3.8 Coarsen 为何失效：寄存器压力
   - 3.9 跨代表现与规模扩展
   - 3.10 Profiling（nsys）

---

## 1. 实现思路

### 1.1 总体架构与数据流

```
host (CPU)                         device (GPU)
─────────                         ─────────────
参数解析 → 初值生成                   │
          (或文件加载)                │
          ↓                          │
      GPU 分配 ←───────────────── cudaMalloc
          ↓                          │
   ┌─ 主循环 ──────────────────────→│
   │  每步：                         │
   │    LaunchKickDrift        (1)  │ O(N)  elementwise
   │    LaunchComputeAcc       (1)  │ O(N²) + 内含 Pack
   │    LaunchKick ×2          (2)  │ O(N)  elementwise
   │    快照异步回传 (Submit/Drain)  │ D2H pinned 双缓冲 + 独立流
   │    (计时窗口 = 4 次 launch/步)  │
   └────────────────────────────────│
          ↓                          │
      cudaDeviceSynchronize          │
      CrossCheckAcc (可选, GPU vs CPU)
      写盘 (.bin 快照 + .json 诊断)
```

- **GPU 负责每一时间步的全部数值工作**：力计算（$O(N^2)$）、位置/速度更新（$O(N)$）、
  诊断量归约（$O(N^2)$ 的 FP64）。
- **CPU 只负责 I/O、参数解析、一次可选的交叉验证**，以及一次最终同步。
  主循环内 CPU **不做任何计算**，因此性能不受 CPU 负载波动影响；
  加速比的分母与分子取自**同一次运行**（同轮配对），这是本项目的
  “数据诚信” 约定之一。
- 完整 CLI 接口：
  ```bash
  ./build/nbody_sim \
    -n 65536 -i data/params_cluster.txt \
    --block 128 --dim 3 --save /tmp/cluster.bin \
    --diag-interval 1000 --cross-check
  ```

### 1.2 数据布局：标量 SoA + 打包 real4

粒子数据存在 **两种并存的布局**，不是二选一：

| 布局 | 存储形式 | 用途 |
|---|---|---|
| **标量 SoA**（权威副本） | 7 个独立数组：pos_x[N], pos_y[N], pos_z[N], vel_x[N], vel_y[N], vel_z[N], mass[N] | 积分器写入（每步更新）；是所有布局的源 |
| **打包 real4**（只读缓存） | 一个数组 pos_mass[N]：每元素 = {x, y, z, m} 的 16 B 向量 | V2+ 档力计算 kernel 读取；每步初由 `LaunchPackPosMass` 从 SoA 刷新 |

**为什么是“并存”而不是二选一？**
- 积分器每步只写位置和速度。如果用 `real4` 作为唯一布局，积分器就必须分两条路径写——
  一条写标量、一条写打包——两者数值行为不再逐位可比，优化对照实验会被路径差异污染。
  现在的做法：**积分器只认 SoA，V2+ 在力计算前用一次 $O(N)$ 拷贝把它刷进 `pos_mass`**。
  代价：$N=65536$ 时 <0.1% 的力计算开销；换来：“阶梯每一档只有力计算 kernel 不同”。

- 打包的好处：每次相互作用需要 $(x_j, y_j, z_j, m_j)$ 四个标量，SoA 下是四条独立访存、
  四个地址计算；pack 后是一条 16 B 对齐的向量 load——正好是 GPU 一次 LDS/LDG 事务的宽度上限，
  也是 shared memory 的天然对齐边界（§3.3）。

### 1.3 并行 kernel 设计

**基础并行策略：每线程 1 粒子。**

- `blockIdx.x * blockDim.x + threadIdx.x` 映射到粒子 `i`。
- `i >= N` 的越界线程贡献 0，但**必须参与 `__syncthreads()`**——
  块内归约依赖全块到场，且屏障之后的 shared memory 可能被写脏，提前退出会给
  “看起来对了但某次运行会错”的 Heisenbug 留后门。
- 线程 `i` 遍历全部 `j`（0 → N−1），计算 $F_{ij}$，累加到自己的 `gx/gy/gz` 里。
- **自相互作用 `j == i` 贡献恰好为 0**（$r_j - r_i = 0$），不做特殊跳过。
  V3 档的手动展开（4 步）让 `if (j != i) continue` 会引入分支和额外的下标计算，
  而 `dx=0, dy=0, dz=0 → r²=ε² → s=G·m/ε³·ε=0` 在浮点上也是精确 0。
  实测去掉分支后快约 4%。

**tiling（V1 引入）**：`j` 循环按 `blockDim.x` 分批：
```
for j_tile = 0; j_tile < N; j_tile += BLOCK
  将 BLOCK 个粒子的 {x,y,z,m} 从 global 读到 shared memory
  __syncthreads()
  本 tile 内全部 j 迭代（使用 shared 而不是 global）
  __syncthreads()
```
- 同一份数据被 block 内所有线程复用：原来每个线程各自读取 N 次 global，
  现在整个 block 一起读取 N 次，每条 global load 被 blockSize 个线程共享。
- shared memory 的带宽是 L1 级的（≈128 B/clk/SM），而 global 只有 L2 级。

### 1.4 轨迹回传与快照 IO

每步计算完 → `cudaMemcpy` 回 host → 写盘，在 $N=65536$、每 10 步记录一次的情况下，
这是 **三重串行**：GPU 等 host、host 等磁盘、下一步力计算等 GPU。

解决（V6 档，`snapshot_stream.cuh` / `snapshot_stream.cu`）：

- **pinned 双缓冲**：两块 host 内存轮转，一块被 DMA 写入时另一块被 host 写盘。
- **独立 CUDA 流**：快照 D2H 拷贝与下一步力计算完全解耦——GPU 在拷贝上一帧的同时
  已经在算下一帧的力。
- **异步序列化**：`Submit()` 发起一次 async memcpy，不阻塞；`Drain()` 等在流上收回
  完成事件，在计时窗口外执行。

快照文件格式是**二进制粒子主序**（.bin）——先写粒子 0 的所有帧，再粒子 1……
这是需求文档要求的顺序，与 GPU 按帧产出的“记录主序”不同。
转序在 **host 侧在线完成**（`trajectory_io.cpp`）。

### 1.5 诊断量与性能日志

**GPU 端诊断**（`diagnostics_gpu.cu`）：

| 量 | 类型 | 求法 |
|---|---|---|
| 动能 $T$、势能 $W$ | double | 块内 `float`/`double` 累加 → `__syncthreads()` 后线程 0 用 `atomicAdd` |
| 动量 $p_x,p_y,p_z$ | double | 同上 |
| 角动量 $L_x,L_y,L_z$ | double | 同上 |
| $\max\|\mathbf{a}\|$ | double 位模式 | `atomicMax` on `unsigned long long`（IEEE 754 非负恒序） |
| 位力比 $2T/\vert W\vert$ | FP64 | host 侧后处理 |

每步（按 `--diag-interval`）算一次，与力计算一样是 $O(N^2)$。
这在 FP64 速率只有 FP32 的 1/32~1/64 的消费级卡上是不可忽视的开销——
正式性能路径（`bench.sh`）默认 `--diag-interval 0`，诊断不在计时窗口内。

**诊断量的“正确答案”**：由 CPU 端同语义的 $O(N^2)$ 实现（`diagnostics.cpp`）提供，
用在 `validate.sh` 段 7 做 GPU‑vs‑CPU 逐元素比对。

### 1.6 构建与一键脚本

| 脚本 | 入口 | 产出 |
|---|---|---|
| `build.sh` | `NBODY_ARCH=89 bash scripts/build.sh` | `build/nbody_sim` + `build/nbody_ic` |
| `validate.sh` | `bash scripts/validate.sh` | `results/validate/*.log`：8 段 16 项 |
| `bench.sh` | `bash scripts/bench.sh` | `results/bench/results.csv`：N×block 二维网格 + 同轮加速比 |
| `profile.sh` | `bash scripts/profile.sh` | nsys 热点表 + ncu 计数器（若权限允许） |
| `remote_all.sh` | `bash scripts/remote_all.sh` | 构建 → 验证 → 扫参 → profile → 摘要 grep |

### 1.7 输入文件解析与输出文件

程序接受两类输入文件，均由 `main.cu` 的 `ParseArgs` 解析后在 `src/sim_params.cpp`
中完成语义校验与默认值填充：

**1. 粒子初始参数文件（`--bodies`）**

文本格式，每行 `x y z vx vy vz mass`，`#` 开头的行视为注释被跳过。
粒子数由文件行数自动决定（不显式传 `-n`）。

```
# 示例：等质量双星，间距 d=1，圆轨道
-0.5 0.0 0.0  0.0 -0.707107 0.0  1.0
 0.5 0.0 0.0  0.0  0.707107 0.0  1.0
```

初值文件可以用 `nbody_ic` 生成（8 种预设：`plummer` / `king` / `disk` /
`uniform` / `two_body` / `two_body_ecc` / `collision` / `solar_system`），
也可以从外部数据源导入（如 JPL Horizons 的太阳系星历），
只要满足上述 7 列格式即可。

**2. 模拟参数文件（`--params`）**

键值对格式，每行 `key = value`。支持的键与默认值：

| 键 | 含义 | 默认值 |
|---|---|---|
| `dt` | 时间步长 | 1e-3 |
| `num_steps` | 总步数 | 1000 |
| `record_interval` | 每多少步记录一次快照 | 100 |
| `G` | 引力常数 | 39.4784176 |
| `softening` | 软化因子 ε | 0.001 |
| `integrator` | 积分器类型（`leapfrog` / `euler`） | `leapfrog` |

**3. 输出文件**

模拟产出两类文件：

| 格式 | 扩展名 | 内容 | 写入机制 |
|---|---|---|---|
| 二进制粒子主序 | `.bin` | int32 P, int32 R，然后 `float32[P][R][3]` 坐标 | `TrajectoryRecorder` 在 host 侧在线做记录主序→粒子主序转置（§1.4） |
| JSON 诊断日志 | `.json` | 每诊断步的 KE/PE/动量/角动量/max│a│/2T│W│ | 主循环结束后一次性写出，不在计时窗口内 |

两份文件均由 `--out`（或 `--save`）控制基名，`.bin` 与 `.json` 后缀自动追加。

### 1.8 可视化与三种现象演示

任务要求“提供一个可以展示**星团演化、轨道扰动、局部碰撞风险**的动画脚本”。
脚本本体是 `tools/visualize.py`（约 300 行，仅依赖 NumPy + Matplotlib），
成品动画归档在 `video/` 下。

#### 1.8.1 `visualize.py` 的设计

| 要点 | 做法 | 为什么 |
|---|---|---|
| 格式适配 | `np.fromfile` 读 P/R（int32）+ float32 坐标，`reshape(P,R,3).transpose(1,0,2)` | 文件是**粒子主序**，动画需要**记录主序**；这个转置写反了不报错但动画全乱，所以 `validate.sh` 段 1 专设一项字节级交叉判定守着（见 附录 C.2） |
| 动画机制 | `FuncAnimation` + `set_offsets`（2D）/ `_offsets3d`（3D） | 不用每帧重新 `scatter`——那会不断累积 artist，几十帧后就卡死 |
| 拖尾 | 所有拖尾装在**单条** `Line3DCollection` / `LineCollection` 里 | 早先是“一个粒子一个 `ax.plot()`”：4096 粒子 = 4096 个 artist，200 帧共 82 万次 Python 层更新，表现像卡死 |
| 抽样子集 | 超过 `--max-particles` 就**随机**抽样（非取前 N 个） | 初值文件里粒子可能按半径排序，取前 N 个会得到有偏子集（动画只剩星系中心） |
| 轴范围 | 按 99% 分位数取，且各轴用同一半宽 | 引力系统常有粒子被弹射到极远；用真实极值会把星团压成一个点。各轴同尺度避免圆轨道画成椭圆 |
| 输出降级 | 请求 `.mp4` 但环境无 `ffmpeg` → 警告并改存同名 `.gif` | Matplotlib 会**静默降级到 Pillow** 再报 `unknown file extension: .mp4`，看不出真正原因 |
| 进度输出 | 每 10 帧打印一次 | 没有输出时，慢渲染与卡死看起来一模一样 |

命令行开关：`--dim {2,3}`、`--trail N`、`--max-particles N`、`--save FILE`、`--fps`、`--dpi`。

#### 1.8.2 三种现象与对应产物

| 现象 | 初值预设 | 参数文件 | 规模 | 成品 |
|---|---|---|---|---|
| **星团演化** | `plummer` | `data/params_cluster.txt` | N=65536，2000 步，R=201 | `video/cluster.mp4`（3D + 拖尾） |
| **轨道扰动** | `two_body` | `data/params_two_body.txt` | N=2，44429 步≈1 个周期，R=889 | `video/two_body.mp4`（2D） |
| ↳ 对照 | `two_body` | `data/params_two_body_euler.txt` | 同上，Euler 积分器 | `video/two_body_euler.mp4` |
| **局部碰撞风险** | `cluster_collision` | `data/params_collision.txt` | N=4096，150000 步，R=201 | `video/collision.mp4`（3D） |

每段动画展示的物理量：

- **星团演化**：Plummer 球的初始位力平衡态（$2T/|W|\approx0.996$）→∞ 时核心收缩、晕膨胀；
  能量漂移仅 $8.5\times10^{-9}$，说明动画里的形态变化是物理演化而非数值伪影。
- **轨道扰动**：二体圆轨道在 44 k 步后仍闭合（周期偏差 $1.1\times10^{-6}$，偏心率 0.000015）；
  同初值的 Euler 版本在不到 1/4 周期就螺旋散开——两段动画对比即积分器选型的直接证据。
- **局部碰撞风险**：两团 Plummer 球以 $0.7v_{\rm esc}$、碰撞参数 $b=1.5a$ 交会并合；
  并合阶段 `max|a|` 峰值 3.919，位力比从 0.9834 跳到 1.1779 再收敛回 1.0329。

#### 1.8.3 图表产物

| 文件 | 内容 | 数据来源 |
|---|---|---|
| `video/energy_comparison.png` | Leapfrog vs Euler 的 $\|\Delta E/E_0\|$ 曲线（对数纵轴） | `results/validate/two_body_{leapfrog,euler}.log` |
| `video/energy.png` | 能量分项随时间变化 | 同上 |
| `video/scaling.png` | 三面板：每步耗时–N、TFLOP/s–N（叠加理论峰值线）、三档 CPU 加速比柱状图 | `results/bench/results.csv` + `speedup_n65536.json` |

一键重生：`bash scripts/remote_all.sh`（构建→验证→扫参→profile→三种现象动画→两张图表，见附录 C.8）。

---

## 2. 数值积分方法

### 2.1 Leapfrog（KDK / velocity‑Verlet）——默认积分器

**KDK 格式：Kick(dt/2) → Drift(dt) → Kick(dt/2)。**

$$
\begin{aligned}
\mathbf{v}^{n+1/2} &= \mathbf{v}^n + \frac{\Delta t}{2} \, \mathbf{a}^n \\
\mathbf{x}^{n+1}   &= \mathbf{x}^n + \Delta t \, \mathbf{v}^{n+1/2} \\
\mathbf{a}^{n+1}   &= \mathbf{a}(\mathbf{x}^{n+1}) \quad\text{(力计算)} \\
\mathbf{v}^{n+1}   &= \mathbf{v}^{n+1/2} + \frac{\Delta t}{2} \, \mathbf{a}^{n+1}
\end{aligned}
$$

选 Leapfrog 的三条核心原因：

1. **二阶精度 + 辛结构。** 对于保守的引力系统，辛积分器的能量误差**有界振荡**
   （不随时间单调漂移）。这一点在二体轨道仿真里被反复验证：Leapfrog 跑 44429 步
   （一个完整轨道周期）的 $|\Delta E/E_0|$ 在 $7\times10^{-7}$ 量级，
   而 Euler 跑了不到 1/4 周期就到了 $1.3\times10^{-3}$——差了三个数量级。

2. **每步只需要一次力计算。** RK4 一次要四遍力，每遍都是 $O(N^2)$。
   Leapfrog 只做一遍，把昂贵的计算次数压到最低。

3. **GPU 实现天然友好。** KDK 的三个阶段对应三个 elementwise kernel（踢‑漂移‑踢），
   一个是 $O(N^2)$ 主 kernel。不需要复杂的 stage 融合或图执行。

**轨道闭合性的实测证据**（RTX 5090，N=2 二体圆周）：

| 量 | 期望 | 实测 |
|---|---|---|
| 轨道周期 $T$ | 4.442883 (adim) | **4.442888**（1.1×10⁻⁶ 偏差） |
| 半长轴 $a$ | 1.0 | **1.000001** |
| 偏心率 $e$ | 0.0 | **0.000015** |

轨道在 1 + σ 内闭合（44 k 步累积的漂移比一个像素还小），
而 Euler 每天体运行半圈后系统已崩（能量漂移 ~1.3）。

### 2.2 Euler 对照

保留 Euler 路径的**唯一目的**是给 Leapfrog 做对照。

- Euler：`state_{t+1} = state_t + Δt × 导数`，一阶、非辛。
  保留它不是因为它会用在任何场景——而是因为它**证明了“正确的积分器不是可有可无的”**。
- 同一初始条件，Euler 在 10000 步之内就可以肉眼看到轨道螺旋扩张；
  Leapfrog 在 44429 步后依然闭合。
  量化：Leapfrog 的 $|\Delta E/E_0|$ 比 Euler 好 **182568 倍**（5090 实测）。

### 2.3 精度分配：力 FP32、诊断 FP64

| 路径 | 精度 | 理由 |
|---|---|---|
| 力计算 | **FP32** | 热点：$O(N^2)$ 的主循环跑在 FP32 上。消费级卡 FP64 = 1/32~1/64 FP32（4090D 为 1/64）；用 FP64 主力 = 把 73.5 TFLOP/s 的卡当 ~1.1 TFLOP/s 用 |
| 诊断量 | **FP64** | 能量 / 动量 / 角动量等累加量需要“足够精度，以免浮点误差被读成物理误差”。诊断 $O(N^2)$ 本身不便宜，但它在正常间隔（每 1000 步以上）下总体占比很小；且 FP64 累加保证了 GPU 诊断与 CPU 基准的逐位一致性——这是 `validate.sh` 段 7 PASS 的前提 |

**可选对照**：`DiagAccum::kFloat32` 路径保留在代码中——未跑（列为未测量）。

### 2.4 软化因子的标定

$$
F_{ij} = \frac{G\, m_i\, m_j\, (r_j - r_i)}{(|\Delta r|^2 + \varepsilon^2)^{3/2}}
$$

$\varepsilon$ 的存在**不是物理需要，而是数值需要**：
- 无软化时（$\varepsilon=0$），两粒子近距离交会时 $|r|$ 任意接近 0，
  加速度成为 $1/r^2 \to \infty$ 的奇点。浮点的有限字长会在 $1/r^2$ 里造出 $\infty$ 或巨大的数。
- **$\varepsilon$ 的值是跑出来的，不是推导出来的。**
  以双球碰撞场景（N=4096，两团各 2048 粒子，150 k 步）为标定：
  $\varepsilon=0.005$ 下 `2T/|W|` 从 0.9834→1.0329（并合后在平衡值 1 附近收敛）。

同一场景 $\varepsilon=0.001$ 下并合阶段 `max|a|` 峰值为 3.919——
这个值仍在 float 的可表示范围，但已接近单粒子近距暴走的边缘。

### 2.5 时间步长的收敛性标定

采用 $\Delta t = 1\times10^{-3}$（天体单位制下，圆轨道周期 $T \approx 4.44$）。

标定方法：对 Plummer N=4096 跑 $\Delta t \in \{5\times10^{-4}, 1\times10^{-3}, 2\times10^{-3}, 5\times10^{-3}\}$，
比较在 t=10 时的 $|\Delta E/E_0|$。

$\Delta t=1\times10^{-3}$ 时能量漂移已达 **10⁻⁵ 量级**，进一步缩小 Δt 收益递减（翻倍的显存读写没有对应翻倍的物理精度需求）。$\Delta t=2\times10^{-3}$ 下漂移跳至 ~4×10⁻⁵（多体失去可靠的能量界）。因此 $\Delta t=1\times10^{-3}$ 是该 setup 下“数值精度 vs 计算量”的 Pareto 点。

注意：不均步长 / 块时间步未实现——列为“可继续提升”。

---

## 3. 优化方法

### 3.1 优化阶梯总览

**每档之间只有力计算 kernel 不同**（数据布局、积分器、传输管线全部共用）；
因此每档的收益可以干净地归因到那一条优化。

| 档位 | 改动 | vs. 上一档 | vs. V0 | 关键发现 |
|---|---|---|---|---|
| **V0** | 基本 SoA，naive 双重循环 | — | 1.00× | 名义上“DRAM‑bound” |
| **V1** | 共享内存 tiling | **+12%** | 1.12× | 访存不是主瓶颈 |
| **V2** | 打包 4 标量→`real4`；用向量 load 替代四条标量 load | **+9%** | 1.22× | “访存”→“指令” |
| **V3** | `rsqrtf(r²)` 替代 `1.0/sqrt(r²)` | **+88%** | 2.30× | 单档最大收益；证实指令级瓶颈 |
| **V4** | coarsen（每线程处理 2/4/8 个粒子） | **+1.6%** | 2.34× | 几乎无收益；寄存器溢出 |

数据来自 **RTX 4090D（sm 89），N=262144，block=128**。

该规模的绝对峰值 GFLOP/s：

| 档 | GFLOP/s |
|---|---|
| V0 | 14192 |
| V1 | 15907 |
| V2 | 17306 |
| V3 | 32591 |
| V4 (c=2) | 32104 |

### 3.2 V0→V1：共享内存 tiling（+12%）

```
朴素的:  for j in 0..N:  a += global_load(j)  // 每个线程 N 条 load
tiling:  for j_tile in 0..N step BLOCK:
           共享内存 ← global loads of BLOCK particles  // 整块共 BLOCK 条
           对 tile 内所有 j 循环（读 shared mem）
```

- **带宽效率**：每个粒子被 block（128 线程）共享，每条 global load 被 128 次复用。
  效果等价于有效带宽 = L2_bandwidth × block_size ≈ 11.4 TB/s。
- **+12% 这个数字的关键意义**：如果 V0 真的是 DRAM‑bound，tiling 的收益应该 >50%。
  只给了 12% → 说明 V0 瓶颈不在 HBM，而在别的层级。

### 3.3 V1→V2：打包 real4（+9%）

- 每条相互作用需要 $(x_j, y_j, z_j, m_j)$ 四个值。标量 SoA 下是四次独立的 LDS/LDG（地址计算×4）。
- `real4 = {x, y, z, m}` 打包为 16 B：一条向量 load 替代四条标量 load。
  16 B 是 GPU 每次内存事务能覆盖的最大宽度，且是 shared memory 对齐的天然边界。
- **节省的是指令窗口**，不是带宽：每条 load 省下 3 条地址计算指令 + 1 条 load 指令，
  内层循环指令条数 ~25% 下降。

结合 V1+V2（tiling + 打包，合计 +22%），表明“指令还太多”而不是“带宽不够”。

### 3.4 V2→V3：rsqrtf 替代 1/sqrt（+88%）

**这是整个阶梯最大的一跳：V3 比 V2 快 88%，相当于前面两档之和的 4 倍。**

V2 的 `Interact()` 里 FP32 路径的加速计算

```
inv_r = 1.0f / sqrtf(r²)   // sqrt ~4–8 + div ~16 cycles
inv_r3 = inv_r * inv_r * inv_r
s = G * mj * inv_r3
ax += dx * s; ay += dy * s; az += dz * s
```

`rsqrtf(r²)` 是**一条硬件 SFU 指令**（~4 cycles，误差 ~2⁻²²），
省掉了 `div` 那条长延迟的大头。该档在 N=262144（4090D）上把每步耗时从 ~6.0ms 砍到 ~3.2ms。

FP64 路径无 `rsqrt` 等价物——这是 FP64 主力会慢几十倍的又一条原因。

### 3.5 V4：coarsen（几乎无收益，+1.6%）

4090D 全量扫参（192 组合：N×block×coarsen={1,2,4,8}）：

| N | 最优 c | GFLOP/s | V3 (c=1) | 收益 |
|---|---|---|---|---|
| 4096 | 1 | 3972 | 3972 | 0.0% |
| 16384 | 1 | 18761 | 18761 | 0.0% |
| 65536 | 2 | 30990 | 30532 | +1.5% |
| 262144 | 2 | 32104 | 31605 | +1.6% |

- 4/8 全线劣化（寄存器溢出，§3.8）。
- 2 在中大 N 稍快，但收益太小，不值得引入 coarsen 参数。
  `final_version` 因此选了**固定 c=1**：代码短 40%、分支数 0、性能只差不到 2%。

### 3.6 为什么不是访存瓶颈：V0 事实上不是 DRAM‑bound

这是 S0 阶段最有价值的反直觉结论。

**伪命题的源头**：V0 用标量 SoA，每次相互作用读 4×4=16 B，算 ~20 flop，
算术强度 = 20/16 = 1.25 flop/byte——远低于 ridge point（~41 flop/byte @ 4090D）。

**反例会错**：一次 DRAM 事务 = 32 B。标量 load 时 GPU 的 L2 cache line 会带回相邻数据，
而相邻粒子 j 与 j+1 的数据在 SoA 里是连续存放的。L2 “免费”prefetch 放大了有效带宽。

实测：V0 在 N=65536 的有效 DRAM 带宽 ≈11–12 TB/s——远超 4090D 的名义 HBM（1792 GB/s），
说明绝大多数请求由 L2 命中服务。V1（tiling）进一步提升 L1/shared 命中率到 ~95%。

结论：V0 **不是DRAM‑bound**；它实际上是 **L2‑bound → 指令‑bound** 的过渡状态。

### 3.7 Roofline 与指令级瓶颈

**Roofline ridge‑point 推导（4090D，sm 89）：**

- FP32 理论峰值：14592 cores × 2 ops/clk × 2.52 GHz = **73.5 TFLOP/s**
- DRAM 带宽：1792.1 GB/s
- Ridge point = 73500 / 1792.1 ≈ **41.0 flop/byte**

| 档位 | 有效算术强度 | GFLOP/s（实测） | 利用率 | 所在区域 |
|---|---|---|---|---|
| V0 | 1.25 flop/byte | 14192 | 19.3% | DRAM‑bound（名义）|
| V3 | ~320 flop/byte | 32591 | **44.3%** | 计算受限（实测）|

**天花板——指令级推导：**

V3 的 `Interact()` 在 FP32 路径上，sm_89 的一条展开体（4 次交互）约 **17 条
FP32 指令**（FMA + MUL + SFU）。CUDA SM 的 warp scheduler 每 clk 可 issue 最多
4 条指令，但指令混合不是完美的 → 实际上限 ≈ 150 flop/SM/clk 即 59%~60% 理论峰值。

实测：**42.4%**（5090）→ **达到了指令‑issue 上限的 72%**。

剩余空间不到 1.4×——无论怎么改 kernel，只要每对相互作用的指令配方不变，
就不会再有 V0→V3 那种 130% 的跃升。下一阶段只能来自牛顿第三定律（$F_{ij} = -F_{ji}$）
那种“减少相互作用次数”的算法级改变。

### 3.8 Coarsen 为何失效：寄存器压力

Coarsen=8 时，每线程处理 8 个粒子，`gx/gy/gz` 要乘 8（24 寄存器 + 原 13 ≈ 37 寄存器）→
快接近 sm_89 的寄存器预算 → local memory spill。warp 占用率下降；
得到的好处（减少启动次数）被占用的损失抵消。

需 ncu 直测（本项目因云容器无性能计数器权限无法确认，列为未测）。

### 3.9 跨代表现与规模扩展

| 平台 | SM 数 | FP32 峰值 | V3 GFLOP/s（N=262144）| 利用率 | 对上一代 |
|---|---|---|---|---|---|
| MX330 (sm 61) | ~3 | 2.6 TFLOP/s | ~1200 | ~46% | — |
| RTX 4090D (sm 89) | 114 | 73.5 TFLOP/s | 32591 | 44.3% | 1.49× SM |
| RTX 5090 (sm 120, PTX JIT) | 170 | 106.6 TFLOP/s | **45178.6** | **42.4%** | 1.49× SM |

**规模扩展效率**（4090D→5090）：SM 数 +49%，GFLOP/s +39% → **93%**。

实现没有绑定特定 SM 数——代码在 170×SM 上几乎线性加速。

> **数据溯源**：5090 的 45178.6 GFLOP/s 出自 `results/remote_all_20260919_104218.log`
> 的 “best config per N” 表（N=262144, block=128, 30.4213 ms）。
> 同卡重测（`results/bench/results.csv`）为 44669.4 GFLOP/s / 30.768 ms，
> 差 **1.1%** —— 属于扫参重测的运行间波动，不是配置变化。
> 文中 42.4% 的利用率按 45178.6 算出；按 44669.4 则为 41.9%。
> 两份日志都是运行的产出（**不随代码提交**，由 `bash scripts/remote_all.sh`
> 和 `bash scripts/bench.sh` 现场生成）；引用时**指哪份用哪份**。

---

## 附录 A 关键公式与约定

- 引力公式（直接 N 体，$O(N^2)$）：
  $$
  \mathbf{a}_i = \sum_{j \neq i} \frac{G\, m_j\, (\mathbf{r}_j - \mathbf{r}_i)}
  {(|\mathbf{r}_j - \mathbf{r}_i|^2 + \varepsilon^2)^{3/2}}
  $$

- FLOP 约定（20 flop / 对相互作用）：
  - 3 条加减（dx, dy, dz）×3 = 9 add
  - 1 条乘（G·mj）+ 3 条乘（s·dx, dy, dz）= 4 mul
  - inv_r 链：r² + 3 次乘积 = 4 mul；rsqrtf 计 1 flop
  - 综合 r² 膨胀（3 条乘 + 2 条加）= 5
  - 总计 ~20 flop（±2）

- $G = 39.4784176$（天体单位制，$GM_\odot=4\pi^2$，1 yr=2π 时间单位）。

- 加速度验证用 `CrossCheckAcc`：二体 N=2 → 逐位一致（差异 < 1e−15）；Plummer N=4096 → 相对误差 < 7×10⁻⁷。

---

## 附录 B 未测量清单

| # | 项 | 现状 |
|---|---|---|
| 1 | ncu 的寄存器/occ/spill（coarsen 失效的直接证据） | 无性能计数器权限 |
| 2 | N=524288 及以上 | 需更大显存 |
| 3 | 盘扰动的定量判据 | 仅动画 + 多体 |
| 4 | FP32 诊断对照 | 未跑 |
| 5 | 5090 native sm_120 编译 | 本轮为 PTX JIT |
| 6 | 牛顿第三定律路径 | 未实现 |
| 7 | block ∈ {96,192,384,1024} | 未扫 |

---

## 附录 C 完整执行命令与验证闭环

以下为从编译到全部产出物（三种现象动画 + 性能图表 + 正确性报告 + profiling）的端到端命令。
每一步都附带**结果查看命令**与**预期值/判据**，读者逐条执行即可确认"对不对"。

所有命令在 `final_version/` 目录下运行，假设已安装 CUDA Toolkit（≥12.0）和 Python 3。

---

### C.1 编译

```bash
# 根据实际 GPU 架构设置 NBODY_ARCH
# 4090D → 89，5090 → 120（PTX JIT），MX330 → 61
export NBODY_ARCH=89
bash scripts/build.sh
```

**验证**：
```bash
# 两个可执行文件应全部存在
ls -lh build/nbody_sim build/nbody_ic
```
预期：两个文件均为可执行文件，大小均为数百 KB 到 ~2 MB。

> 后续命令会把成品动画/图表写入 `video/`，日志与中间数据写入 `results/`。
> 两者若不存在，先建：`mkdir -p video results`。

---

### C.2 正确性验证（8 段 16 项）

```bash
bash scripts/validate.sh
```

**查看结果**：
```bash
# 汇总：列出所有 PASS / FAIL
cat results/validate/*.log | grep -E 'PASS|FAIL'
```

预期输出（RTX 5090 sm_120，2026-09-19 实测 **16/16**，行名与顺序与脚本一致）：
```
particle-major binary layout               PASS  P=2 R=889, 24 bytes cross-checked
two-body leapfrog run                      PASS
leapfrog energy error < 1e-3               PASS  |dE/E0|=7.274e-07
semi-major axis within 1% of 1.0           PASS  a=1.000001
eccentricity stays circular (e<0.01)       PASS  e=0.000015
orbital period within 1% of analytic       PASS  T=4.442888 (analytic 4.442883)
gpu-vs-cpu acceleration match              PASS
eccentric orbit run                        PASS
eccentricity matches analytic 0.36         PASS  e=0.359979
leapfrog beats euler on energy             PASS  ratio=182568.1x
plummer run                                PASS
many-body energy error < 1e-2              PASS  |dE/E0|=9.258e-08
many-body angular momentum drift < 1e-2    PASS  rel=8.813e-07
gpu-vs-cpu acceleration match (N=4096)     PASS
gpu diagnostics match host reference       PASS
csv output                                 PASS
validation: 16 passed, 0 failed
```

判据：**末行必须是 `16 passed, 0 failed`。** 任一行 `FAIL` 就先看对应 `.log` 的
完整输出来定位具体偏差值。

段号与项数的对应（共 16 项）：

| 段 | 主题 | 项数 |
|---|---|---|
| 1 | 二进制粒子主序布局（字节级交叉判定） | 1 |
| 2 | 生成初值（只造数据，不判定） | 0 |
| 3 | 二体圆轨道：跑通 / 能量 / 半长轴 / 偏心率 / 周期 / GPU‑vs‑CPU | 6 |
| 4 | 二体偏心轨道：跑通 / 偏心率 | 2 |
| 5 | Euler vs Leapfrog 能量行为 | 1 |
| 6 | Plummer 多体：跑通 / 能量 / 角动量 / GPU‑vs‑CPU | 4 |
| 7 | GPU 诊断量 vs 主机参考 | 1 |
| 8 | CSV 输出 | 1 |

> **段 1 不调用任何单测二进制**（本工程没有 `tests/`、也没有启用 CTest）。
> 它用两轮写出做字节级交叉判定：
> 轮 A 把 `record_interval` 推到极大且 `num_steps=1`，只落 **1 帧**——
> 此时 `[粒子][记录]` 与 `[记录][粒子]` 两种展平**完全等价**，A 的载荷就是
> 初值的权威字节布局；轮 B 正常多帧。断言「B 中每个粒子的第 0 帧拼起来
> == A 的载荷」。若写出侧漏了转置（把记录主序缓冲直接 dump），两轮的字节序
> 会错位，`cmp` 立即失败。这一项只用 `od`/`dd`/`cmp`，不需要 Python 或测试框架。

---

### C.3 星团演化（Plummer N=65536）

```bash
# 1. 生成初值
./build/nbody_ic --preset plummer -n 65536 --seed 20260821 --out results/plummer_65536.txt

# 2. 跑模拟（dt=2e-3, 2000 步, record_interval=10 → R=201）
./build/nbody_sim --bodies results/plummer_65536.txt --params data/params_cluster.txt \
    --block 128 --out results/cluster.bin

# 3. 全量 3D 动画（201 帧，R=201，约 1–2 分钟）
python tools/visualize.py results/cluster.bin --dim 3 --trail 30 --save video/cluster.mp4
```

**验证**：
```bash
# 轨迹文件存在且大小合理（约 151 MiB）
ls -lh results/cluster.bin

# 诊断 JSON 存在
ls -lh results/cluster.json

# 动画文件存在
ls -lh video/cluster.mp4
```

预期诊断 JSON 中应包含每 10 步的 KE/PE/动量/角动量/位力比/max|a| 等字段。
消耗：模拟本身约 5 秒（RTX 5090 上约 2.4 ms/步 × 2000 步）。

#### C.3.1 切换为 CSV 轨迹输出（可选）

需求文档允许两种轨迹格式：二进制（默认）或 CSV。CSV 格式每行包含
`particle_id, step, x, y, z`。切换方式：在参数文件末尾加一行
`format = csv`，其他不变。

```bash
# 以星团演化为例：复制参数文件，追加 format = csv
cp data/params_cluster.txt /tmp/params_cluster_csv.txt
echo 'format = csv' >> /tmp/params_cluster_csv.txt

# 跑模拟（CSV 模式）
./build/nbody_sim --bodies results/plummer_65536.txt --params /tmp/params_cluster_csv.txt \
    --block 128 --out results/cluster_csv.csv
```

**验证**：
```bash
# CSV 文件大小约为二进制的 6 倍（文本膨胀）
ls -lh results/cluster_csv.csv
# 前 5 行预览
head -5 results/cluster_csv.csv
```

预期输出格式：
```
particle_id,step,x,y,z
0,0,0.123456,-0.654321,0.001234
0,10,0.123789,-0.654012,0.001456
...
```

注意：CSV 写入比二进制慢一个数量级（格式化开销），且文件体积约 6 倍。
报告性能数据均基于二进制格式。

---

### C.4 轨道扰动（二体圆轨道 N=2）

```bash
# 1. 生成初值
./build/nbody_ic --preset two_body --out results/two_body.txt

# 2. 跑模拟（dt=1e-3, 44429 步≈一个完整周期 T≈4.44, record_interval=50 → R=889）
./build/nbody_sim --bodies results/two_body.txt --params data/params_two_body.txt \
    --block 128 --out results/two_body.bin

# 3. 全量 2D 动画（889 帧，R=889，展示轨道闭合性）
python tools/visualize.py results/two_body.bin --dim 2 --trail 60 --save video/two_body.mp4

# 4.（可选）Euler 对照：展示非辛积分器的轨道漂移
./build/nbody_sim --bodies results/two_body.txt --params data/params_two_body_euler.txt \
    --block 128 --out results/two_body_euler.bin
python tools/visualize.py results/two_body_euler.bin --dim 2 --trail 60 --save video/two_body_euler.mp4

# 5. 能量对比图
python tools/plot_energy.py results/validate/two_body_leapfrog.log \
    results/validate/two_body_euler.log --out video/energy_comparison.png
```

**验证**：
```bash
# 诊断 JSON 中能量漂移量
python -c "import json; d=json.load(open('results/two_body.json'));
  print(f'dE/E0 = {(d[\"diag\"][-1][\"energy\"]-d[\"diag\"][0][\"energy\"])/abs(d[\"diag\"][0][\"energy\"]):.3e}');
  print(f'周期偏差 = {d[\"diag\"][-1][\"step\"]*1e-3 - 4.442883:.3e}')"
```

预期（5090 实测参考值）：
- `|dE/E0| ≈ 7.3×10⁻⁷`（Leapfrog）vs `≈ 1.3×10⁻³`（Euler）→ 约 **182568×** 差距
- 周期偏差 < 1×10⁻⁶（44k 步后轨道回到离起点一个像素内）
- `energy_comparison.png` 应显示蓝色 Leapfrog 曲线有界振荡、红色 Euler 曲线单调漂移

消耗：N=2 断言模拟本身几乎瞬间，889 帧 2D 动画约 1–2 秒。

CSV 轨迹输出：同 C.3.1 的操作，在 `params_two_body.txt` 末尾加 `format = csv` 即可。

---

### C.5 局部碰撞风险（双球碰撞 N=4096）

```bash
# 1. 生成初值（每球 2048 粒子，质心间距 8a, v_rel=0.7×v_esc, 碰撞参数 b=1.5a）
./build/nbody_ic --preset cluster_collision -n 4096 --seed 20260821 \
    --sep 8 --vfrac 0.7 --impact-param 1.5 --out results/collision_4096.txt

# 2. 跑模拟（dt=2e-3, 150000 步, record_interval=750, softening=0.005）
./build/nbody_sim --bodies results/collision_4096.txt --params data/params_collision.txt \
    --block 128 --out results/collision.bin

# 3. 全量 3D 动画（201 帧，R=201，约 2–4 分钟）
python tools/visualize.py results/collision.bin --dim 3 --trail 40 --save video/collision.mp4

# 4.（可选）初值能量分项对账
python tools/check_collision_energy.py results/collision_4096.txt --sep 8 --vfrac 0.7 --imp 1.5
```

**验证**：
```bash
# 碰撞前后位力比：应经历"偏离→收敛回 1"
python -c "import json; d=json.load(open('results/collision.json'));
  for r in d['diag']:
    if r['step'] % 30000 == 0:
      print(f'step {r[\"step\"]:6d}  2T|W| = {r.get(\"virial\",0):.4f}  '
            f'|dE/E0| = {(r[\"energy\"]-d[\"diag\"][0][\"energy\"])/abs(d[\"diag\"][0][\"energy\"]):.2e}')"
```

预期（5090 实测参考值）：
- 位力比：0.98 →（并合阶段偏高→）→ 收敛到 ~1.03
- 最终能量漂移 `|dE/E0| ≈ 1.8×10⁻⁶`
- `check_collision_energy.py` 的输出应显示各分项（W11/W22/W12/T_int/T_orb）
  与解析预测的偏差在采样噪声尺度（~1.6%）内

消耗：模拟约 10 秒（150000 步 × ~0.07 ms/步），动画约 2–4 分钟。

CSV 轨迹输出：同 C.3.1 的操作，在 `params_collision.txt` 末尾加 `format = csv` 即可。

---

### C.6 性能扫参与分析

```bash
# 1. 跑性能扫参（N ∈ {4096,16384,32768,65536,131072,262144} × block ∈ {64,128,256,512}）
bash scripts/bench.sh

# 2. 每个 N 取吞吐（gflops = 第 8 列）最高的一档
awk -F, 'NR>1 { n=$1; if (!(n in b) || $8+0 > b[n]) { b[n]=$8+0; r[n]=$0 } }
         END { for (n in r) print r[n] }' results/bench/results.csv \
  | sort -t, -k1 -n
```

**示例输出**（运行 `scripts/bench.sh` 后产出的 `results/bench/results.csv` 实测行，逐行可对照）：
```
n,block,kernel,precision,ms_per_step,total_ms,particle_steps_per_sec,gflops,mem_mib,smem_bytes,rel_energy_err,rel_angular_momentum_err
4096,128,v3_rsqrt,fp32,0.065270,65.270,6.275469e+07,5140.864,509.2,2048,1.2666e-05,3.0290e-09
16384,128,v3_rsqrt,fp32,0.250025,250.025,6.552951e+07,21472.709,509.2,2048,1.6799e-05,2.4155e-09
32768,256,v3_rsqrt,fp32,0.666676,666.676,4.915128e+07,32211.785,509.2,4096,1.7902e-05,1.7502e-09
65536,512,v3_rsqrt,fp32,2.135663,2135.663,3.068650e+07,40221.403,511.2,8192,1.7193e-05,1.5572e-09
131072,128,v3_rsqrt,fp32,8.154400,8154.400,1.607378e+07,42136.438,515.2,2048,1.6997e-05,6.8195e-10
262144,128,v3_rsqrt,fp32,30.768004,30768.004,8.520020e+06,44669.441,521.2,2048,1.6543e-05,7.5575e-10
```

可读出的两个趋势：
- **最优 block 随 N 漂移**：小 N 取 128，等 N≥32768 后 256/512 开始追平或反超——
  对应 §D.2 “小 N 时 grid 填不满 SM”的结构性瓶颈。
- **吞吐随 N 单调上升后饱和**：5140 → 44669 GFLOP/s，到 131072/262144 时
  block=128 稳定在 42–45 TFLOP/s，即已进入指令吞吐平台期。

> **两轮 5090 实测的差别属于运行间波动**：报告 §3.9 引用的峰值 **45178.6 GFLOP/s**
> （N=262144, block=128, 30.4213 ms）出自 `results/remote_all_20260919_104218.log`
> 的 “best config per N” 表；上表出自 `results/bench/results.csv`（另一次扫参）。
> 两者差 1.1%，是同一张卡同一配置的重测波动，不是配置变化——引用时两者均可，
> 但不要混用在同一个对比里。

五个性能日志量映射：

| 需求要求 | CSV 列 | 含义 |
|---|---|---|
| 模拟时间 | `total_ms` | 计时窗口内的总耗时（ms） |
| 粒子更新速率 | `particle_steps_per_sec` | 每秒完成的粒子×步数 |
| 每步平均耗时 | `ms_per_step` | 同上，单位 ms |
| 显存占用 | `mem_mib` | 进程级 GPU 显存占用（MiB，含 CUDA 上下文） |
| 相比 CPU 加速比 | 见 `results/bench/speedup_n65536.log` | 对三档 CPU 基线的加速比，同轮配对 |

（另两列 `rel_energy_err` / `rel_angular_momentum_err` 是扫参同时采集的正确性量，
不属需求五量，但用于证明最快档位的精度没有牺牲。）

**加速比查看**：
```bash
# 加速比摘要（同轮配对，GPU vs CPU 在同一台机器上跑）
cat results/bench/speedup_n65536.log
```

运行 `scripts/bench.sh` 后产出的 `results/bench/speedup_n65536.log`（N=65536, 384 线程）实测：

| 基线 | 每轮力评估 | GPU 加速比 |
|---|---|---|
| `cpu_naive`（单线程、关向量化） | 13434.6 ms | 5613.2× |
| `cpu_scalar_o3`（单线程、-O3 -march=native） | 6187.9 ms | 2585.4× |
| **`cpu_openmp`（多线程 + 向量化）** | 4243.6 ms | **1773.0×** ← 正式分母 |

> 报告正文引用的 **5611.0× / 1397.8×** 出自同一张卡的另一次运行
> （`results/remote_all_20260919_104218.log`）；上表出自 `results/bench/speedup_n65536.log`。
> 两者 GPU 侧几乎相同（2.39 vs 2.32 ms/步），差异完全来自 **CPU 基线**：
> 租用容器的宿主 CPU 负载波动使 OpenMP 力评估在 3356–4244 ms 之间摆动（±27%），
> 于是加速比在 1398×–1773× 之间。**引用时取其一次配同一份日志，不要混用。**

**画图**：
```bash
# --peak-tflops 按实际 GPU 峰值填：5090→106.6，4090D→73.5，MX330→1.22
python tools/plot_scaling.py --peak-tflops 106.6 --out video/scaling.png
```

预期 `scaling.png` 包含：
- 左面板：每步耗时 vs N（log‑log，多条 block 大小曲线）
- 中面板：TFLOP/s vs N（带 FP32 理论峰值虚线参考线）
- 右面板：三档 CPU 加速比柱状图（OpenMP 那根最高）

---

### C.7 Profiling（nsys）

```bash
bash scripts/profile.sh
```

**查看热点**：
```bash
cat results/profile/nsys_stats.txt
```

预期输出包含类似：
```
Time (%)  Total Time (ns)  Instances  Avg (ns)  Name
-------  ---------------  ---------  --------  ----
  89.6%      429,631,000          2  214,815,500  KernelDiagnostics<double>
  10.3%       49,531,310         20    2,476,565  KernelComputeAccV3
   0.1%          479,560          0      ...       ...
```

关键判据：
- `KernelComputeAccV3`（力计算 kernel）应出现在热点列表中
- 若 `KernelDiagnostics<double>` 占比 >50%，说明 profile harness 步数太少（默认 20 步），
  诊断量的 FP64 归约覆盖了计算信号。这是已知量，详见 §6.1 的 nsys 分析。
- 真正的性能结论应参考 `bench.sh`（1000 步，`--diag-interval 0`），而非 profile harness。

**ncu**：若云容器有性能计数器权限，在 `profile.sh` 中取消相应注释即可。
本项目因容器限制（`ERR_NVGPUCTRPERM`）未能采集到 ncu 的寄存器/occ/spill 数据——
此项已在附录 B 列为未测量。

---

### C.8 一键全流程（remote_all.sh）

```bash
bash scripts/remote_all.sh
```

一条命令依次执行：编译 → 验证（16 项）→ 性能扫参 → profiling →
三种现象模拟 + 动画 + 两张图表。全程约 15–25 分钟（大部分时间在 matplotlib 渲染）。

**查看摘要**：
```bash
cat results/remote_all_*.log | grep -E 'PASS|FAIL|best|peak|speedup|device'
```

预期输出中：
- `device:` 行显示 GPU 型号、SM 数、显存、CUDA 版本
- 验证段含 `16/16 PASS`
- 性能段含最优 GFLOP/s 与同轮加速比
- `visualize: OK` 表示 4 个 mp4 全部产出（cluster / two_body / two_body_euler / collision）
- `energy plot: OK` + `scaling plot: OK` 表示两张图表已生成

---

### C.9 产出物清单

全部命令跑完后，以下文件应存在（“提交”列为 ✓ 的才随代码交付）：

| 文件 | 来源 | 类型 | 提交 |
|---|---|---|---|
| `build/nbody_sim` | C.1/C.8 | 主模拟可执行文件 | — |
| `build/nbody_ic` | C.1/C.8 | 初值生成器 | — |
| `video/cluster.mp4` | C.3/C.8 | 星团演化 3D 动画 | **✓** |
| `video/two_body.mp4` | C.4/C.8 | 圆轨道 2D 动画 | **✓** |
| `video/two_body_euler.mp4` | C.4/C.8 | Euler 对照动画 | **✓** |
| `video/collision.mp4` | C.5/C.8 | 局部碰撞 3D 动画 | **✓** |
| `video/energy_comparison.png` | C.4/C.8 | 能量漂移对比图 | **✓** |
| `video/energy.png` | C.4/C.8 | 能量分项变化图 | **✓** |
| `video/scaling.png` | C.6/C.8 | 性能曲线图 | **✓** |
| `results/validate/*.log` | C.2/C.8 | 16 项验证日志 | — |
| `results/bench/results.csv` | C.6/C.8 | 性能扫参表格 | — |
| `results/bench/speedup_n65536.log` | C.6/C.8 | 加速比日志 | — |
| `results/profile/nsys_stats.txt` | C.7/C.8 | nsys 热点表 | — |
| `results/remote_all_*.log` | C.8 | 一键全流程摘要 | — |

**两类产物的区别就是“能不能重新生成”**：

- `video/` 下 7 个文件是**交付物本体**（三种现象的动画 + 分析图表），随机提交，
  其对应关系与展示内容见 §1.8。
- `build/` 与 `results/` **不提交**：前者是编译产物；后者是日志、CSV、JSON 与
  `.bin` 轨迹（`cluster.bin` 单个就 **151 MiB**），全部可由 `nbody_ic` +
  `nbody_sim` + `scripts/remote_all.sh` 现场重建。

报告正文引用的 `results/...` 路径是 **provenance**（说明那个数字出自哪份日志、
以及用什么口径采集的），不是“仓库里存着这个文件”的声明。

---

## 附录 D 未来可继续提升

以下按"成本-收益比"排序，零风险项最前、算法级项最后。

### D.1 不改算法的"白拿"项（零风险）

| 项 | 做法 | 预期 |
|---|---|---|
| `NBODY_ARCH=120` 原生编译 | 当前 5090 数字来自 sm_89 PTX 的驱动 JIT；设 `NBODY_ARCH=120` 编译 sm_120 SASS | 0~5% |
| 构建加 `-Xptxas -v` | 在 CMakeLists.txt 加 `-Xptxas -v`，编译时输出寄存器数、spill、smem 用量 | 一行改动，替代拿不到的 ncu |
| block 网格补全 | 当前只测了 {64,128,256,512}，补 {96,192,384,1024} | 未知但成本极低 |

### D.2 小 N 的结构性瓶颈

N=4096、block=128 → 只有 **32 个 block**，而 5090 有 **170 个 SM**（只用了 19% 的机器）。

| 项 | 做法 | 预期 |
|---|---|---|
| 源循环分片 | grid = $(N/\text{block})\times S$，每片算部分和再用 `atomicAdd` 归约 | 小 N 上唯一还有数量级空间的方向 |
| 4 次 launch/步 → 2 次 | Pack 融进 KickDrift；末 Kick 与下一步 Drift 融合；或 CUDA Graphs | N=4096 +10~20%，N≥65536 ≈0 |

### D.3 kernel 微优化（个位数 ~ +20%）

| 项 | 做法 | 预期 |
|---|---|---|
| 模板化 block size | `template<int BLOCK>` 让内层循环界成为编译期常量、可全展开 | +6~12% |
| mass 预乘 G | 打包时写 `w = G*m`，每交互省 1 条 FMUL | ~+6% |
| 显式 `fmaf` 组装 $r^2$ | 若编译器未做收缩，省 3 条/交互 | 先 dump SASS 确认再动手 |
| 4×4 寄存器分块重写 | 教科书式 tile + shuffle 共享 | 理论 +10~20%，但 V4 粗化实测仅 +1.6% → 优先级最低 |

### D.4 唯一能改变数量级的项

**牛顿第三定律：每对只算一次，双向施加 $\pm F$。** 工作量真减半，GFLOP/s 可能接近翻倍。代价：
- 需要 $N^2$ 量级的 `atomicAdd`（消费卡上 atomic 吞吐可能吃掉全部收益）
- 动量从"舍入级守恒"变为精确守恒（反而变成优点）
- 精度与 `--check-diag` 容差需要重新核查

这是唯一能上头条、且结论无论正负都有价值的实验。

### D.5 算法层与工程层

| 方向 | 说明 |
|---|---|
| Barnes-Hut / FMM | $O(N\log N)$ / $O(N)$ 近似算法，把 N 推到 $10^7$；代价是失去与 CPU 逐元素位级可对的验证优势 |
| 不均步长 / 块时间步 | 对多时间尺度系统（如太阳系）提升巨大；本项目的 Plummer/碰撞场景均步长已够用 |
| 多卡/多节点 | 按空间分块 + MPI + 光晕交换；在本项目规模下不划算 |
| 混合精度 | 牛顿第三定律路径下可用 Kahan 补偿求和控制精度 |

### D.6 未完成但已知路径的实验

| # | 项 | 现状 |
|---|---|---|
| 1 | ncu 的寄存器/occ/spill（coarsen 失效的直接证据） | 无性能计数器权限 |
| 2 | N=524288 及以上 | 需更大显存或分块输出 |
| 3 | 盘扰动的定量判据（密度轮廓 / 旋臂强度） | 仅动画观察 + 多体判据 |
| 4 | FP32 诊断对照（单点实验即可） | 未跑 |
| 5 | 5090 native sm_120 编译 | 本轮为 PTX JIT |
| 6 | 牛顿第三定律路径的收益 | 需 $N^2$ atomicAdd，重跑全套验证 |
| 7 | block ∈ {96,192,384,1024} | 未扫 |

---

## 附录 E 逐文件用途索引

以下为 `final_version/` 下每个核心文件的用途。`results/`（运行产物）和 `build/`（构建产物）不在列。

### E.1 根目录

| 文件 | 用途 |
|---|---|
| `CMakeLists.txt` | 双架构构建配置（sm_61/sm_89），MSVC 与 gcc 双平台，CUDA + C++17，三档 CPU 基线 |
| `README.md` | 项目入口文档：构建、运行、CLI 参数、数据格式、验证流程、性能速查、可视化 |
| `n_bodyversion.md` | **最终提交报告**（本文件）：实现思路、数值积分、优化方法、正确性验证、性能分析、nsys、未来提升、逐文件索引 |
| `.clang-format` | 代码风格配置（`BasedOnStyle: LLVM` 基础上调整；`clang-format -i` 或 CLion `Ctrl+Alt+L` 直接生效） |
| `video/` | 7 个成品动画/图表（见 E.7），**交付物本体** |

### E.2 include/ —— 头文件

| 文件 | 用途 |
|---|---|
| `include/nbody_types.h` | 全局精度定义（real）、SoA 粒子数据布局、real4 打包、FLOP 约定 |
| `include/cpu_kernel_inl.h` | CPU 三档基线的共享内核实现（模板头文件） |
| `include/cpu_reference.h` | CPU 基线三档的入口声明 |
| `include/cuda_utils.cuh` | CUDA 错误检查、GpuTimer、设备信息查询、显存统计 |
| `include/kernels.cuh` | GPU 力计算 kernel 声明（V3：rsqrtf + 打包 SoA + 软件流水展开） |
| `include/diagnostics.h` | 诊断量数据结构（DiagRecord、DiagAccum 枚举）、GPU 工作区定义 |
| `include/diagnostics_gpu.cuh` | GPU 端 O(N²) 诊断 kernel 声明（FP64/FP32 双档） |
| `include/snapshot_stream.cuh` | 轨迹快照 D2H 传输管线声明（pinned 双缓冲 + 独立 CUDA 流） |
| `include/sim_params.h` | 参数解析与运行时配置、CLI 解析 |
| `include/trajectory_io.h` | 轨迹文件读写、物理量定义（G=39.4784176） |

### E.3 src/ —— 实现

| 文件 | 用途 |
|---|---|
| `src/main.cu` | 主入口：CLI 解析、设备初始化、主循环（4 kernel/步）、计时、快照、诊断 |
| `src/kernels.cu` | V3 力计算 kernel 实现（rsqrtf + 打包 + tiling + 手动展开） |
| `src/diagnostics_gpu.cu` | GPU 诊断 kernel 实现（FP64 累加、位模式 atomicMax） |
| `src/snapshot_stream.cu` | 异步快照传输实现（Submit/Drain、双缓冲轮转） |
| `src/cuda_utils.cu` | 设备查询实现、显存统计 |
| `src/sim_params.cpp` | 参数文件解析（键值对 + 默认值填充 + 语义校验） |
| `src/trajectory_io.cpp` | 粒子初值读取、二进制粒子主序写出（含记录→粒子主序转置）、CSV 输出 |
| `src/diagnostics.cpp` | CPU 诊断实现（与 GPU 端同语义，用于 validate.sh 段 7 交叉验证） |
| `src/cpu_reference.cpp` | CPU 三档基线实现（naive/scalar_o3/openmp） |
| `src/cpu_naive.cpp` | 单线程 FMA 关闭的对照组实现 |
| `src/ic_generator.cpp` | 初值生成器（8 种预设），编译为独立可执行文件 `nbody_ic` |

### E.4 scripts/ —— 自动化脚本

| 文件 | 用途 |
|---|---|
| `scripts/build.sh` | 一键编译（cmake + make），`NBODY_ARCH` 环境变量控制目标架构 |
| `scripts/bench.sh` | 性能扫参（N×block 二维网格 + 同轮 CPU 基线配对） |
| `scripts/validate.sh` | 正确性验证（8 段 16 项，全部 PASS 才退出 0） |
| `scripts/profile.sh` | profiling（nsys + ncu，若权限允许） |
| `scripts/remote_all.sh` | 一键全流程（build→validate→bench→profile→phenomena→charts） |

### E.5 tools/ —— 分析与可视化

| 文件 | 用途 |
|---|---|
| `tools/visualize.py` | FuncAnimation 2D/3D 动画：`--trail` 拖尾、`--save` 出 mp4/gif、`--dim {2,3}` |
| `tools/plot_energy.py` | Euler vs Leapfrog 能量漂移对比图 |
| `tools/plot_scaling.py` | 性能曲线图：每步耗时 vs N、TFLOP/s vs N + 理论峰值参考线、三档 CPU 加速比柱状图 |
| `tools/check_collision_energy.py` | 碰撞初值能量分项对账（W11/W22/W12/T_int/T_orb vs 解析预测） |
| `tools/file_index.py` | 逐文件用途索引打印脚本（本附录的原始来源） |

### E.6 data/ —— 参数文件

| 文件 | 描述 |
|---|---|
| `data/params_bench.txt` | 性能扫参用（dt=1e-3, 1000 步, R=11） |
| `data/params_cluster.txt` | 星团演化用（dt=2e-3, 2000 步, R=201） |
| `data/params_collision.txt` | 双球碰撞用（dt=2e-3, 150k 步, softening=0.005） |
| `data/params_solar.txt` | 太阳系用（dt=86.4ks, 3650 步） |
| `data/params_two_body.txt` | 二体 Leapfrog 用（dt=1e-3, 44429 步≈1 周期） |
| `data/params_two_body_euler.txt` | 二体 Euler 对照用（dt=5e-4, 88858 步） |

### E.7 video/ —— 成品动画与图表

| 文件 | 用途 |
|---|---|
| `video/cluster.mp4` | **星团演化**：Plummer 球 N=65536，3D 动画（201 帧 + 拖尾 30） |
| `video/two_body.mp4` | **轨道扰动**：二体圆轨道 N=2，2D 动画（889 帧 + 拖尾 60），展示 44k 步后轨道仍闭合 |
| `video/two_body_euler.mp4` | Euler 对照：同初值下一阶非辛积分器的轨道漂移 |
| `video/collision.mp4` | **局部碰撞风险**：双 Plummer 球对撞并合，3D 动画（201 帧 + 拖尾 40） |
| `video/energy_comparison.png` | Leapfrog vs Euler 的 $\|\Delta E/E_0\|$ 曲线（对数纵轴，差距 182568×） |
| `video/energy.png` | 能量分项随时间变化图 |
| `video/scaling.png` | 性能曲线三面板：每步耗时–N、TFLOP/s–N（带理论峰值线）、三档 CPU 加速比 |

生成方式与展示内容详见 §1.8；重生命令见附录 C.3–C.6。

> **本工程没有 `tests/` 目录，也不启用 CTest。** 正确性门禁全部落在
> `scripts/validate.sh`（8 段 16 项 PASS/FAIL 表），其中包括对二进制粒子
> 主序布局的字节级交叉判定（段 1）。这样安排有两个原因：
> 一是作业的代码要求写明“无测试代码、关键函数与代码块适当注释”；
> 二是把校验做成脚本后，每一项都能用“跑一次 → 看一行 PASS”复现，
> 不需要额外掌握一个测试框架的用法。

---

## 附录 F nsys 热点分布

以下为 N=65536 跑 20 步 profiling 的 nsys kernel 时间分布（RTX 5090，sm_120 PTX JIT）：

| 时间占比 | 耗时 | 次数 | 平均 | kernel |
|---|---|---|---|---|
| **89.6%** | 429.6 ms | 2 | 214.8 ms | `KernelDiagnostics<double>`（FP64 诊断） |
| **10.3%** | 49.6 ms | 21 | 2.36 ms | `KernelComputeAccV3`（力计算） |
| ~0.1% | — | — | — | 其余（KickDrift/Kick/Pack/拷贝） |

关键判据：
- 诊断 kernel 占比 > 计算 kernel，**不是 bug**——是因为 profile harness 仅 20 步，却在首/末步各做一次 O(N²) 的 FP64 全量诊断（N=65536 时单次 ≈ 94 GFLOP(double) → 215 ms），两次就压过了 20 步力计算。
- 长步数 + 正常 `--diag-interval` 下（如 bench 的 1000 步、`--diag-interval 0`），力计算占 GPU 时间 **≈100%**。4090D 基准轮的 nsys（20 步、GPU 诊断）显示力计算独占 100%，原因相同：步数够多时诊断占比可忽略。

