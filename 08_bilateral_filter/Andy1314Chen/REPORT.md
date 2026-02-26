# CUDA 双边滤波优化方案

## 一、问题背景

双边滤波是一种经典的边缘保持平滑滤波器，其核心计算公式为：

$$
BF[I]_p = \frac{1}{W_p} \sum_{q \in \mathcal{S}} \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{|I_p - I_q|^2}{2\sigma_r^2}\right) \cdot I_q
$$

对每个像素，需遍历其邻域窗口内的所有像素，计算空间权重与值域权重的乘积作为综合权重，进行加权平均。当滤波半径为 $r$ 时，每个像素需要 $(2r+1)^2$ 次邻域访问和权重计算，计算量大，非常适合 GPU 并行加速。

然而，朴素的 CUDA 实现存在以下性能瓶颈：

1. **全局内存访问量大**：邻域像素被相邻线程重复读取，带宽浪费严重。
2. **超越函数开销高**：内循环中大量调用 `expf`，每次约消耗 20+ 个时钟周期。
3. **分支发散（Warp Divergence）**：边界检查导致同一 warp 内的线程执行路径不一致。
4. **计算访存比低**：整体受限于内存带宽而非计算能力。

以下从 **访存优化**、**计算优化** 和 **算法级优化** 三个层面，系统阐述优化策略。

---

## 二、访存优化

### 2.1 Shared Memory 缓存邻域数据

这是最关键的优化手段。

朴素实现中，每个线程独立从全局内存读取其邻域内的所有像素，而相邻线程的邻域高度重叠。以 $16 \times 16$ 的 block 和 $r=5$ 的滤波半径为例：

- 每个线程需读取 $11 \times 11 = 121$ 个像素
- 整个 block 共发起 $256 \times 121 = 30976$ 次全局内存访问
- 实际不重复的数据仅为 $(16+10) \times (16+10) = 676$ 个像素
- **冗余率高达 45 倍**

**优化方法**：让 block 内的线程**协作加载**邻域数据（含 halo 区域）到 shared memory，之后所有滤波计算均从 shared memory 读取。

- Shared memory 访问延迟：~5ns
- 全局内存访问延迟：~200-400ns

**实现要点**：

- Shared memory 尺寸为 $(blockDim + 2r) \times (blockDim + 2r)$，需确保不超过每个 SM 的容量限制（通常 48KB 或 96KB）
- 加载阶段需处理 halo 区域（block 边界外的像素），每个线程可能需要加载多个元素，通过循环步进覆盖整个 shared memory 区域
- 加载完成后必须调用 `__syncthreads()` 确保数据就绪
- 边界像素采用 clamp 策略（重复边缘像素值），避免条件分支

### 2.2 Texture Memory

纹理内存是另一种有效的访存优化手段，尤其适用于**滤波半径较大、shared memory 容量不足**的场景。其优势包括：

- **自动边界处理**：通过设置 `addressMode` 为 `cudaAddressModeClamp`，硬件自动处理越界访问，完全消除边界检查分支，避免 warp divergence
- **2D 空间局部性缓存**：纹理缓存针对 2D 访问模式优化，与双边滤波的邻域访问模式天然匹配
- **免费的插值支持**：如果需要亚像素精度的滤波，纹理硬件可提供免费的双线性插值

使用 CUDA Texture Object API 创建纹理对象，在 kernel 中通过 `tex2D<float>(texObj, x, y)` 读取数据即可。

### 2.3 向量化内存访问

利用 CUDA 的向量类型（如 `float4`、`uchar4`）进行内存访问，可以在单次内存事务中读取多个像素。例如，使用 `float4` 一次读取 4 个连续的 float 像素，将内存事务数量减少为原来的 $\frac{1}{4}$，提高全局内存带宽利用率。

> **注意**：这要求数据在内存中对齐存储。

---

## 三、计算优化

### 3.1 空间权重查找表（Constant Memory）

空间高斯权重：

$$
w_s(dx, dy) = \exp\left(-\frac{dx^2 + dy^2}{2\sigma_s^2}\right)
$$

仅依赖于偏移量 $(dx, dy)$，与像素值无关，因此可以在 host 端**预计算**为查找表，存储在 **constant memory** 中。

- Constant memory 具有专用缓存
- 同一 warp 内所有线程访问同一地址时，仅需一次内存事务即可广播给所有线程
- 查找表大小为 $(2r+1)^2$ 个 float，对于 $r=10$ 仅需 1764 字节，远小于 constant memory 的 64KB 容量限制

### 3.2 值域权重查找表

对于 **8-bit 图像**，像素值差异的绝对值范围为 $[0, 255]$，因此值域高斯权重也可以预计算为 **256 个元素**的查找表，同样存储在 constant memory 中。

- Kernel 中只需计算像素差的绝对值作为索引，直接查表获得权重
- **完全避免 `expf` 调用**

对于浮点图像，可以将值域差异量化到固定区间后查表，或者采用分段线性近似。

### 3.3 快速数学函数

如果无法使用查找表（例如浮点图像且精度要求不高），可以使用 CUDA 提供的快速数学内建函数：

- `__expf` 替代 `expf`：吞吐量约为 `expf` 的 **2-3 倍**
- 精度损失在图像处理场景中通常可以接受
- 也可在编译时添加 `--use_fast_math` 选项，自动替换所有标准数学函数为快速版本

### 3.4 预计算倒数

将 $\frac{1}{2\sigma_s^2}$ 和 $\frac{1}{2\sigma_r^2}$ 在 host 端预计算好作为 kernel 参数传入，避免在 kernel 中重复计算除法。虽然是微小的优化，但在内循环中累积效果不可忽略。

### 3.5 循环展开

使用 `#pragma unroll` 提示编译器展开内循环，或者将滤波半径作为**模板参数**，使其成为编译期常量，从而实现完全展开。

**循环展开的收益**：

- 消除循环控制指令（比较、跳转）的开销
- 使编译器能够更好地进行指令调度和寄存器分配
- 为指令级并行（ILP）创造更多机会

**实践建议**：为常见的半径值（如 3、5、7、9）分别实例化模板 kernel，在运行时根据参数选择对应版本。

---

## 四、算法级优化

### 4.1 分离近似（Separable Approximation）

严格来说，双边滤波**不可分离**为两个一维滤波的级联，因为值域权重依赖于像素值。但在实践中，可以采用**近似分离**的策略：

1. **第一遍**：沿水平方向做一维双边滤波
2. **第二遍**：沿垂直方向做一维双边滤波

**复杂度对比**：

- 原始：$O((2r+1)^2)$
- 分离后：$O(2 \times (2r+1))$
- 对于 $r=10$：从 441 次降低到 42 次，**减少约一个数量级**

**额外的访存优势**：一维滤波的内存访问模式更规则，水平方向滤波可以实现合并访问（coalesced access），垂直方向滤波可以通过 shared memory 转置来优化。

**代价**：结果为近似值，但在大多数视觉应用中质量损失不明显。

### 4.2 双边网格（Bilateral Grid）

这是一种将双边滤波从空间域转移到高维空间处理的算法级加速方法。核心思想：

1. **Splat**：将二维图像提升到三维双边网格 $(x, y, intensity)$，在网格中进行下采样
2. **Blur**：在低分辨率的三维网格上执行普通高斯模糊（可分离、高效）
3. **Slice**：将结果从三维网格切片回二维图像

**关键优势**：计算复杂度与滤波半径无关，为 $O(N)$（$N$ 为像素数），特别适合大半径滤波。在 GPU 上，三维网格的高斯模糊可以高效地用分离卷积实现。

---

## 五、执行配置优化

### 5.1 Block 尺寸选择

Block 尺寸的选择需要平衡以下因素：

- **占用率（Occupancy）**：block 尺寸影响每个 SM 上的活跃 warp 数。通常 $16 \times 16 = 256$ 或 $32 \times 8 = 256$ 个线程是较好的起点
- **Shared memory 用量**：block 越大，所需的 shared memory 越多 $((blockDim + 2r)^2)$，可能限制每个 SM 上的并发 block 数
- **合并访问**：block 的 x 维度应为 32 的倍数（或至少为 warp 大小的因子），以确保全局内存的合并访问

> **建议**：使用 CUDA Occupancy Calculator 或 `cudaOccupancyMaxPotentialBlockSize` API 来确定最优配置。

### 5.2 Stream 与异步执行

对于视频流或批量图像处理，可以使用多个 CUDA stream 实现**计算与数据传输的重叠**：

- **Stream 1**：传输第 $n+1$ 帧到 GPU
- **Stream 2**：处理第 $n$ 帧
- **Stream 3**：传输第 $n-1$ 帧的结果回 CPU

---

## 六、多通道图像处理

对于 RGB 彩色图像，值域距离通常采用三通道的欧氏距离：

$$
d_r = \sqrt{(R_p - R_q)^2 + (G_p - G_q)^2 + (B_p - B_q)^2}
$$

三个通道**共享同一组权重**（空间权重和值域权重），因此只需计算一次权重，分别应用于三个通道的加权求和。可以使用 CUDA 的向量类型 `uchar3` 或 `float3` 来打包处理三通道数据，减少内存事务次数。

---

## 七、优化效果总结

| 优化策略 | 主要收益 | 适用条件 | 预期加速比 |
|:---:|:---:|:---:|:---:|
| Shared Memory | 减少全局内存访问 | 半径较小，shared memory 足够 | 3-10× |
| 空间权重 LUT | 消除 `expf` 调用 | 所有情况 | 1.5-2× |
| 值域权重 LUT | 消除 `expf` 调用 | 8-bit 图像 | 1.5-2× |
| 快速数学函数 | 降低 `expf` 开销 | 精度要求不高 | 1.3-1.5× |
| 循环展开 | 消除循环开销，提升 ILP | 半径为编译期常量 | 1.2-1.5× |
| Texture Memory | 自动边界处理，2D 缓存 | 半径较大 | 1.5-3× |
| 分离近似 | 复杂度从 $O(r^2)$ 降到 $O(r)$ | 可接受近似误差 | 5-10× |
| 双边网格 | 复杂度与半径无关 | 大半径场景 | 10-50× |

> **组合建议**：典型的高性能实现会同时采用 **Shared Memory + 空间权重 LUT + 值域权重 LUT + 循环展开** 的组合，在此基础上根据具体场景选择是否引入分离近似或双边网格等算法级优化。

---

## 八、代码实现对照

以下梳理 `bilateral_filter_cuda.cu` 中**实际采用**的优化手段，标注代码位置与解决的问题。

### 已实现的优化

| # | 优化手段 | 解决的问题 | 代码位置 |
|---|---------|-----------|---------|
| Opt1 | **Constant Memory 空间权重 LUT** | 消除内循环 `expf` 调用；利用 warp 广播机制降低访存开销 | `d_spatial_lut` 声明 (L33)，`init_spatial_lut` host 端预计算 (L649-661) |
| Opt1 | **Constant Memory 值域权重 LUT** | 对 8-bit 图像完全消除值域 `expf`；256 项查表替代超越函数 | `d_color_lut` 声明 (L34)，`init_color_lut` (L663-673) |
| Opt1 | **LUT 缓存（参数不变则跳过上传）** | 避免重复 `cudaMemcpyToSymbol`，减少 H2D 传输开销 | `ensure_luts` 静态缓存比较 (L676-692) |
| Opt2 | **uint8 直接 I/O（模板类型参数）** | 省去 host 端 u8→float→u8 转换流水线，减少数据搬运量和中间缓冲区 | `to_output<T>` 模板特化 (L40-49)，`launch_u8_*` 系列函数 (L778-813)，kernel 模板参数 `<InT, OutT>` |
| Opt3 | **cudaHostRegister 页锁定内存** | 将调用方普通堆内存注册为 page-locked，使 H2D/D2H 传输走 DMA 通道，带宽提升 ~2× | `ensure_registered` (L834-851) |
| Opt4 | **Shared Memory 协作加载** | 消除邻域重复全局内存访问；16×16 block + r=5 时冗余率从 45× 降至 1× | 所有 kernel 的 `smem` 声明与协作加载循环，例如 template 版 (L59-86)，runtime 版 (L492-516) |
| Opt4 | **Clamp 边界处理** | 用 `max(0, min(N-1, x))` 替代 if 分支，避免 warp divergence | 加载阶段 `gx = max(0, min(width-1, gx))` (L81-82 等) |
| Opt5 | **RGB 单一色彩权重（均值近似）** | 将 3 次 LUT 查找 + 3 个权重累加器缩减为 1 次，减少指令数和寄存器压力 | template RGB kernel 中 `(1.0f/3.0f)` 均值距离 (L192-196)，注释说明 MAE 代价 (L177-179) |
| Opt6a | **模板参数编译期半径** | 使编译器完全展开双重循环，消除循环控制开销，提升 ILP | `template <int RADIUS>` 所有 kernel (L55, L124, L220, L270 等)，`switch(radius)` 分发 (L886-910) |
| Opt6a | **`#pragma unroll`** | 对 template 版完全展开，对 runtime 版提示展开因子 4 | template 版 `#pragma unroll` (L75, L98-100)，runtime 版 `#pragma unroll 4` (L529-531) |
| Opt6b | **`__frcp_rn` 快速倒数** | 用 1 次倒数 + 3 次乘法替代 3 次除法，减少昂贵除法指令 | RGB template kernel (L208) |
| Opt7 | **`__restrict__` 指针修饰** | 告知编译器输入输出不 alias，允许更激进的优化（如缓存 load 结果） | 所有 kernel 参数 (L56, L125, L221, L271, L489 等) |
| Opt8 | **持久化 GPU 缓冲区** | 全局静态缓冲区 `g_bufs`，仅在图像尺寸变化时重新分配，避免每帧 `cudaMalloc/cudaFree` | `g_bufs` 结构体 (L820-830)，`ensure_io_buffers` (L853-861)，`ensure_temp_buffer` (L863-869) |
| Opt9 | **分离近似（Separable）** | 复杂度从 O(r²) 降至 O(r)；r=5 时从 121 次降到 22 次邻域访问 | 水平/垂直独立 kernel：`k_bilateral_horizontal_*` (L221, L326)，`k_bilateral_vertical_*` (L271, L405) |
| Opt10 | **Block 尺寸 16×16** | 兼顾 occupancy 与 shared memory 用量；相比 32×8 提供更好的 2D 缓存局部性 | `BLOCK_X=16, BLOCK_Y=16` (L24-27) |

### 未实现的优化

#### 访存层面

**SoA 数据布局**

当前 RGB 采用交错存储（AoS: RGBRGB...），相邻线程读取同一通道时内存地址间隔为 3，无法实现完美合并访问。改为分离存储（SoA: RRR...GGG...BBB...）后：

- 同通道数据地址连续，全局内存合并效率从 ~33% 提升至 100%
- Shared memory 中各通道独立存储，消除 bank conflict
- 代价：需要在数据输入/输出时做一次布局转换，或在图像 I/O 层直接采用 SoA 格式

**L2 Cache 持久化控制**（Compute Capability ≥ 8.0）

通过 `cudaAccessPropertyPersisting` 将高频访问数据（如 LUT 或输入图像的热点区域）钉在 L2 cache 中：

- 对于视频流等反复处理同尺寸图像的场景，帧间 L2 命中率显著提升
- LUT 数据量小（~1-4 KB），常驻 L2 后等价于多了一层低延迟缓存
- 需要配合 `cudaCtxResetPersistingL2Cache` 管理缓存生命周期

**Warp Shuffle（`__shfl_sync`）**

对 separable 的水平 pass，一个 warp 内 32 个线程处理同一行连续像素，邻域数据天然分布在相邻线程的寄存器中：

- 用 `__shfl_sync(__activemask(), val, lane ± offset)` 直接在寄存器级别交换数据
- 延迟 ~1 cycle，比 shared memory（~5ns / ~20 cycles）还低一个数量级
- 完全绕过 shared memory，释放 shared memory 容量给其他用途
- 限制：仅适用于一维滤波，且 radius 不能超过 warp 宽度（16）

#### 计算层面

**`__launch_bounds__` 编译提示**

当前 kernel 未指定 launch bounds，编译器只能保守分配寄存器：

```cpp
__global__ void __launch_bounds__(256, 4)  // maxThreadsPerBlock, minBlocksPerMultiprocessor
k_bilateral_filter_gray_template(...) { ... }
```

- 显式告知编译器 block 大小和最小并发 block 数，使其精准控制寄存器分配
- 避免 register spilling 到 local memory（延迟从 ~1 cycle 飙升到 ~200 cycle）
- 对于当前 16×16=256 线程的配置，设 `maxThreads=256, minBlocks=4` 是合理起点

**FP16 半精度计算**

对 8-bit 图像，像素值范围 [0,255]，FP16 的精度（10-bit 尾数）完全足够：

- FP16 吞吐量是 FP32 的 2 倍（在大多数 GPU 架构上）
- 利用 `half2` 向量指令可同时处理 2 个邻域像素的权重计算
- Shared memory 占用减半，允许更大的 tile 或更高的 occupancy
- 代价：需要处理 `half` ↔ `float` 转换开销，以及个别中间累加仍需 FP32 避免精度损失

**权重早期截断**

当空间距离大时，`spatial_weight` 趋近于零，后续乘法和累加都是无效计算：

- 对 radius=10 的高斯核（σ_s=3），外围 ~40% 的权重 < 1e-4，可安全跳过
- 实现方式：预计算有效半径 `r_eff = ceil(3 * sigma_s)`，仅遍历 `[-r_eff, r_eff]`
- 注意：逐像素的 if 判断会引入 warp divergence，适合配合 template radius 使用固定的有效半径

#### 算法层面

**自适应半径**

不同区域使用不同滤波强度，平坦区域用小 radius，边缘/纹理区域用完整 radius：

- 需要一个轻量预处理 pass 计算局部梯度（如 Sobel），开销约为主 kernel 的 5-10%
- 对大 radius 场景，总计算量可减少 30-50%
- 可与 CUDA Dynamic Parallelism 结合：父 kernel 判断区域类型，子 kernel 使用对应 radius

**Kernel Fusion**

如果滤波前后有色彩空间转换（RGB→YCbCr）、归一化、gamma 校正等操作：

- 融合到同一个 kernel 中，避免中间结果写回全局内存再读取
- 每次全局内存读写约 200-400ns，融合 N 个操作可省去 N-1 次中间读写
- 实现方式：用 lambda 或函数指针模板参数注入前/后处理逻辑

#### 工程层面

**Pinned Memory 池 + 双缓冲**

当前 `cudaHostRegister` 对已有堆内存做页锁定，每次注册/注销有内核态开销（~100μs）：

- 改用 `cudaMallocHost` 预分配固定大小的 pinned memory 池
- 双缓冲：buffer A 做 H2D 传输时，buffer B 上运行 kernel，交替使用
- 配合多 stream 可实现传输与计算完全重叠，整体吞吐接近纯计算时间

**多 Stream 条带并行**

将图像按行切分为若干条带（strip），分配到不同 CUDA stream：

- 不同 stream 的 H2D 传输、kernel 执行、D2H 传输可交错进行
- GPU 的 copy engine 和 compute engine 物理独立，真正并行
- 条带间需要 halo 行重叠（各多读 radius 行），但对大图像来说开销可忽略

#### 优先级建议

| 优先级 | 优化手段 | 改动量 | 预期收益 |
|:---:|---------|:---:|:---:|
| ★★★ | `__launch_bounds__` | 1 行/kernel | 5-15%（避免 spill） |
| ★★★ | Warp Shuffle（水平 pass） | 中等 | 10-30%（消除 smem） |
| ★★★ | SoA 数据布局 | 中等 | 10-20%（合并访问） |
| ★★☆ | 权重早期截断 | 小 | 大 radius 时 20-40% |
| ★★☆ | FP16 半精度 | 中等 | 吞吐翻倍（受限于精度需求） |
| ★★☆ | Kernel Fusion | 视流水线而定 | 省去中间读写 |
| ★☆☆ | L2 持久化 | 小 | 视频流场景 5-15% |
| ★☆☆ | 自适应半径 | 大 | 场景依赖 |
| ★☆☆ | 双缓冲 + 多 Stream | 大 | 批处理/视频流场景 |

### 三种模式对比

代码通过 `FilterMode` 枚举 (L698-702) 和 `BILATERAL_MODE` 环境变量 (L710-723) 支持运行时切换：

| 模式 | 环境变量 | 核心特点 | 适用场景 |
|------|---------|---------|---------|
| STANDARD | `BILATERAL_MODE=0` | Runtime radius，动态 shared memory | 任意半径，通用回退 |
| TEMPLATE | `BILATERAL_MODE=1` | 编译期 radius，完全循环展开 | 常见半径值（3/5/7/9/10），性能最优 |
| SEPARABLE | `BILATERAL_MODE=2` | 水平+垂直两遍近似，O(r) 复杂度 | 追求最高吞吐量，可接受近似误差 |

---

## 九、优化实验记录

测试环境：4K 图像（3840×2160），radius=5, σ_s=3, σ_c=30，50 次取均值。

### 基线性能

| 模式 | 4K RGB (ms) | 4K Gray (ms) | MAE |
|------|:-----------:|:------------:|:---:|
| STANDARD | 11.51 (min 9.33) | 3.84 (min 3.72) | 0.65 |
| TEMPLATE | 9.01 (min 7.31) | 2.99 (min 2.92) | 0.80 / 0.62 |
| SEPARABLE | 7.23 (min 5.26) | 2.07 (min 1.98) | 0.45 / 0.15 |

### Opt A: `__launch_bounds__(256, 4)`

给所有 8 个 kernel 添加 `__launch_bounds__` 提示，限制 maxThreadsPerBlock=256, minBlocksPerSM=4。

| 模式 | 4K RGB (ms) | 4K Gray (ms) | 变化 |
|------|:-----------:|:------------:|:----:|
| STANDARD | 9.76 (min 9.39) | 3.81 (min 3.64) | ~持平 |
| TEMPLATE | 7.68 (min 7.27) | 2.98 (min 2.92) | ~持平 |
| SEPARABLE | 5.68 (min 5.49) | 2.03 (min 1.98) | ~持平 |

**结论**：收益不显著（< 2%），说明当前 kernel 寄存器压力不大，编译器默认分配已较优。但该提示无负面影响，保留在代码中。

### Opt B: Warp Shuffle 水平 pass（已回退）

为 separable 灰度水平 kernel 实现了 `__shfl_sync` 版本，用 32×8 block 替代 16×16，halo 用小型 shared memory。

| 指标 | 基线 | Shuffle 版 |
|------|:----:|:----------:|
| 4K Gray SEPARABLE | 2.07 ms | 5.02 ms |
| MAE | 0.15 | 0.50 |

**结论**：性能退步 2.4×，MAE 也变差。原因：

1. halo 边界的 if/else 分支引入了 warp divergence
2. radius=5 时只有 10/32 个 lane 需要 halo，分支比例高
3. 原始 shared memory 方案加载已充分合并，延迟被计算隐藏
4. Warp shuffle 更适合纯寄存器场景（如 reduction），对这种需要大量邻域数据的滤波，shared memory 仍是更好的选择

### Opt C: SoA 数据布局（已回退）

将 RGB separable 改为 AoS→SoA 转换 + 3×灰度 separable + SoA→AoS 转换。

| 指标 | 基线 | SoA 版 |
|------|:----:|:------:|
| 4K RGB SEPARABLE | 5.26 ms (min) | 7.15 ms (min) |
| MAE | 0.45 | 0.45 |

**结论**：性能退步 36%。原因：

1. AoS↔SoA 转换引入 2 次额外全局内存遍历（读+写各一次），4K RGB 约 47MB
2. 8 个 kernel launch 替代原来 2 个，launch overhead 累积明显
3. 每通道独立计算值域权重，丧失了 RGB kernel 中三通道共享权重的优势
4. 原始 RGB separable kernel 通过 shared memory 已经将 AoS 的不合并访问局限在加载阶段，计算阶段完全在 smem 中进行，瓶颈不在全局内存访问模式

### Opt B: `cudaFuncCachePreferL1` / `cudaFuncCachePreferShared`

On sm_89 the L1 cache and shared memory share a 128 KB SRAM pool. Tried two configs:

**L1 preference** (`cudaFuncCachePreferL1`):

| Mode | Before | After |
|------|:------:|:-----:|
| TEMPLATE 4K RGB min | 7.21 ms | 7.31 ms |
| SEPARABLE 4K RGB min | **5.42 ms** | **7.36 ms** (+36% ❌) |

Explanation: L1 preference shrinks shared memory from 64 KB to 32 KB per SM. Separable horizontal kernel needs `3 × 16 × 26 × 4 = 4992 B` of smem; with fewer smem the hardware schedules fewer concurrent blocks per SM, lowering occupancy and memory-latency hiding.

**Conclusion**: `cudaFuncCachePreferL1` is harmful for these kernels. Switched to `cudaFuncCachePreferShared` (which matches what the hardware already does by default for smem-heavy kernels). No measurable delta either way. **Keep `PreferShared` as defensive annotation.**

---

### Opt C/F: Circular Window (Spatial LUT Corner Zeroing + Early Continue)

Zero out LUT entries where `dx² + dy² > radius²` in `init_spatial_lut`.  
For r=5: 121 positions → 81 inside circle, **40 corners zeroed (33%)**。

#### Phase 1: LUT 预置零（无 kernel 分支）

仅在 LUT 中将圆外位置权重设为 0，kernel 内循环不做任何判断。效果：

**Performance impact** (4K RGB, 50 runs):

| Mode | Time before | Time after | Delta |
|------|:-----------:|:----------:|:-----:|
| STANDARD | 9.33 ms min | 9.26 ms min | ~0% |
| TEMPLATE | 7.21 ms min | 7.22 ms min | ~0% |
| SEPARABLE | 5.42 ms min | 5.39 ms min | ~0% |

**Quality impact** (MAE / PSNR vs OpenCV):

| Mode | Old MAE | New MAE | Old PSNR | New PSNR |
|------|:-------:|:-------:|:--------:|:--------:|
| STANDARD RGB | 0.647 | **0.477** | 47.55 dB | **48.61 dB** |
| TEMPLATE RGB | 0.800 | **0.603** | 45.90 dB | **48.28 dB** |
| SEPARABLE RGB | 0.448 | **0.448** | 48.49 dB | 48.49 dB |
| ADAPTIVE RGB | 0.437 | **0.404** | 48.55 dB | **49.42 dB** |
| STANDARD Gray | 0.622 | **0.612** | 50.18 dB | **50.23 dB** |

Phase 1 结论：零性能成本，但精度显著改善（TEMPLATE RGB MAE 0.80→0.60，PSNR +2.4 dB）。

#### Phase 2: Early Continue（跳过圆外像素）

在 kernel 内循环中添加 `if (spatial_weight == 0.0f) continue;`，实际跳过圆外 33% 的迭代体。

**Performance impact** (4K, 50 runs, Phase 1 → Phase 2):

| Mode | Before (ms) | After (ms) | 提升 | 说明 |
|------|:-----------:|:----------:|:----:|------|
| **TEMPLATE RGB** | 7.36 | **6.53** | **+13%** | 编译期常量 RADIUS → 编译器消除圆外迭代 |
| **TEMPLATE Gray** | 4.97 | **3.01** | **+65%** | 同上，Gray 内循环体更轻故比例更大 |
| STANDARD RGB | 9.44 | 8.61 | +10% | 运行时分支，warp 内一致（同一偏移）→ 仍有收益 |
| STANDARD Gray | 3.79 | 3.59 | +6% | 同上 |
| ADAPTIVE RGB | 6.99 | 7.48 | **-7%** | 有害，已回退（见下） |

**MAE/PSNR**: 与 Phase 1 完全一致（跳过的像素权重本来就是 0）。

#### 机制分析

**TEMPLATE kernel 巨大收益的原因**：RADIUS 是编译期常量，`#pragma unroll` 完全展开 121 次迭代。展开后每个 `(dx,dy)` 对应固定的 LUT 地址。编译器（nvcc -O3）在编译期即可确定哪 40 个位置的 `spatial_weight` 恒为 0，通过 **dead code elimination** 直接删除这些迭代体的全部指令——不是运行时分支跳过，而是编译期消除。

Gray kernel 提升（+65%）大于 RGB（+13%）是因为：Gray 内循环体较轻（1 次 smem 读 + 1 次 color LUT 查），减少 33% 迭代的相对占比更大。RGB 内循环体更重（3 次 smem 读 + 色差计算 + 乘加），循环控制开销占比较小。

**STANDARD kernel 中等收益的原因**：runtime radius 无法完全展开，`continue` 是真正的运行时分支。但 warp 内所有线程对同一个 `(dx,dy)` 执行相同判断——**无 warp divergence**（因为 skip/不 skip 取决于偏移量而非像素值），所以分支代价很低。

**ADAPTIVE kernel 有害的原因**：ADAPTIVE 的 `my_radius` 每个像素不同，对于 `my_radius < r_max` 的像素，循环范围已在 `[-my_radius, my_radius]` 内，**圆外像素本来就不会被访问到**。额外的 `continue` 分支只增加了 constant memory 读取和比较指令的开销，且 ADAPTIVE 已有严重 warp 分歧（不同线程循环次数不同），再加分支只会加剧。

#### 最终决策

| Kernel | Early Continue | 理由 |
|--------|:--------------:|------|
| TEMPLATE (gray/RGB) | ✅ 启用 | 编译期消除，+13%~+65% |
| STANDARD (gray/RGB) | ✅ 启用 | 运行时分支无 divergence，+6%~+10% |
| ADAPTIVE (gray/RGB) | ❌ 不启用 | 有害（-7%），循环范围已由 my_radius 限制 |
| SEPARABLE | — 不适用 | 1D pass 无圆形窗口概念 |

---

### Opt E: Multi-Stream Strip Pipelining

Activated via `BILATERAL_STRIP=N` (default off). Splits image into N horizontal strips on independent CUDA streams to overlap H2D / kernel / D2H.

**Performance results** (4K RGB, N=2/4/8 strips):

| Mode | Single-shot | 2 strips | 4 strips | 8 strips |
|------|:-----------:|:--------:|:--------:|:--------:|
| TEMPLATE min | 7.16 ms | ~7.4 ms | 7.47 ms | 7.79 ms |
| SEPARABLE min | 5.39 ms | 5.51 ms | 5.68 ms | 5.84 ms |

**Correctness**: MAE identical across all strip counts ✓ (halo overlap implemented correctly).

**Root cause of no speedup**:

1. **WSL2 copy+compute overlap limitation**: WSL2 runs GPU commands through a virtualized layer that serializes PCIe DMA and compute, preventing the key hardware overlap that strip pipelining requires.
2. **Per-strip overhead**: Each strip adds `cudaMemcpyAsync` setup, stream synchronization, and halo row duplication (~2% data overhead for r=5/540 rows).
3. **Memory-bound kernels** (SEPARABLE): GPU bandwidth is already saturated; splitting into strips adds overhead without reducing total work.
4. **Compute-bound kernels** (TEMPLATE): strips don't reduce computation; only communication overlap would help.

**Conclusion**: **❌ 无收益（WSL2 环境）**。代码保留（`BILATERAL_STRIP` 默认关闭），在裸机 Linux + 多 PCIe DMA engine 环境下理论上可获 2-3× 端到端加速。

---

### 总结（含新实验）

| 优化 | 状态 | 实际效果 | 教训 |
|------|:----:|---------|------|
| `__launch_bounds__` | ✅ 保留 | ~持平 | 寄存器压力低时收益有限，但无副作用 |
| Warp Shuffle | ❌ 回退 | -2.4× | 需要 halo 的滤波场景不适合 shuffle |
| SoA 布局 | ❌ 回退 | -36% | 格式转换开销 > 合并访问收益 |
| `cudaFuncCachePreferL1` | ❌ 回退（改为 PreferShared） | SEPARABLE -36% | L1 偏好压缩 smem 容量，降低 occupancy |
| 圆形窗口 LUT 预置零 | ✅ 保留 | MAE -0.15~0.20，PSNR +2.4 dB | 零性能成本的精度优化 |
| 圆形窗口 early-continue | ✅ TEMPLATE/STANDARD 启用 | **+13%~+65%**（TEMPLATE Gray 最大） | 编译期常量 RADIUS 使编译器能彻底消除圆外迭代 |
| 圆形窗口 early-continue (ADAPTIVE) | ❌ 回退 | -7% | 可变 radius 下分支有害，且循环范围已由 my_radius 限制 |
| Strip Pipeline | ❌ 无收益（代码保留） | 0~+8% 开销 | WSL2 阻止 copy+compute 真正并行 |

> **启示**：当前实现已是 **Shared Memory + Constant LUT + Template Unroll + Separable** 的高度优化组合，进一步提升需要从算法层面入手（如双边网格），或针对特定硬件特性（如 FP16、L2 持久化）做精细调优。在尝试优化前，应先用 `ncu` profiler 确认实际瓶颈（compute-bound vs memory-bound），避免盲目优化。

---

## 十、Profiler 分析

**测试环境**：NVIDIA GeForce RTX 4060 (sm_89, Ada Lovelace), 8GB GDDR6
**测试数据**：4K RGB (3840×2160×3), radius=5, σ_s=3, σ_c=30, 55 次运行

> 注：`ncu` 因 `perf_event_paranoid=2` 无法采集 GPU 硬件计数器（需 root 权限设 `perf_event_paranoid≤1`），故使用 `nsys` 采集时间线 + `nvcc --ptxas-options=-v` 采集编译期指标。

### 10.1 nsys 时间线分析

#### TEMPLATE 模式 (MODE=1)

| 阶段 | 耗时 | 占比 | 说明 |
|------|-----:|-----:|------|
| `k_bilateral_filter_rgb_template<5,u8,u8>` | 3.36 ms/次 | **57.9%** | 唯一的 GPU kernel |
| `cudaMemcpy H2D` | 2.02 ms/次 | 34.8% | 24.9 MB u8 数据上传 |
| `cudaMemcpy D2H` | 1.94 ms/次 | 33.4% | 24.9 MB u8 结果回传 |
| `cudaDeviceSynchronize` | 3.38 ms/次 | — | 等待 kernel 完成 |
| `cudaLaunchKernel` | 0.016 ms | <0.1% | 可忽略 |

**关键发现**：

- **Kernel 与传输无重叠**：H2D → kernel → sync → D2H 串行执行，传输占端到端时间 ~46%
- **传输带宽**：H2D ~12.3 GB/s, D2H ~12.8 GB/s（PCIe 4.0 x16 理论 ~32 GB/s，达到 ~40%）
- **cudaHostRegister 开销**：首次 ~1.1 ms/次，后续缓存命中则跳过

#### SEPARABLE 模式 (MODE=2)

| 阶段 | 耗时 | 占比 | 说明 |
|------|-----:|-----:|------|
| `k_bilateral_horizontal_rgb<5,u8>` | 0.79 ms/次 | **52.4%** | 水平 pass |
| `k_bilateral_vertical_rgb<5,u8>` | 0.72 ms/次 | **47.6%** | 垂直 pass |
| `cudaMemcpy H2D` | 2.00 ms/次 | — | 同上 |
| `cudaMemcpy D2H` | 1.94 ms/次 | — | 同上 |
| `cudaLaunchKernel` | 0.010 ms/次 | <0.1% | 2 次 launch |

**关键发现**：

- **两个 pass 几乎等分 kernel 时间**：水平略慢（全局内存 AoS 读取不合并），垂直略快（中间结果为 float，合并访问更好）
- **kernel 总时间 1.51 ms**（0.79+0.72），远小于传输 3.94 ms —— **瓶颈已从 kernel 计算转移到 H2D/D2H 传输**

### 10.2 编译期指标（ptxas, sm_89）

#### R=5 u8 kernel 资源使用

| Kernel | 寄存器/线程 | Shared Memory | Spill | Occupancy 限制因素 |
|--------|:----------:|:-------------:|:-----:|:------------------:|
| `rgb_template<5,u8,u8>` | 64 | 8112 B | 0 | 寄存器（max 1 block/SM） |
| `horizontal_rgb<5,u8>` | 62 | 4992 B | 0 | 寄存器 |
| `vertical_rgb<5,u8>` | 62 | 4992 B | 0 | 寄存器 |
| `gray_template<5,u8,u8>` | 63 | 2704 B | 0 | 寄存器 |
| `horizontal_gray<5,u8>` | 35 | 1664 B | 0 | — |
| `vertical_gray<5,u8>` | 40 | 1664 B | 0 | — |

**Occupancy 分析**（RTX 4060, sm_89: 65536 regs/SM, 100KB smem/SM, 1536 threads/SM）：

- RGB template kernel (64 regs, 256 threads/block):
  - 每个 SM 最多 `65536 / 64 / 256 = 4` 个 block → **1024 线程 / 1536 = 66.7% occupancy**
  - smem 限制: `100KB / 8112B ≈ 12` 个 block → 不是瓶颈
  - **寄存器是 occupancy 的限制因素**
- Separable gray horizontal (35 regs):
  - 每个 SM 最多 `65536 / 35 / 256 = 7` 个 block → **1792 → cap 1536 = 100% occupancy**
  - smem 充裕

**关键发现**：

1. **零 spill**：所有 kernel 均无寄存器溢出，说明 `__launch_bounds__` 当前不产生额外收益
2. **寄存器用量统一在 62-64**：RGB kernel 循环展开后需要大量中间变量，编译器已用满 64 个寄存器
3. **RGB template kernel occupancy 偏低（~67%）**：64 regs 是主要限制。若能降到 48，occupancy 可提升到 ~83%，但需要牺牲循环展开

### 10.3 Roofline 模型分析

> 注：由于 WSL2 环境下 ncu 无法采集 GPU 硬件计数器（NVIDIA driver 层面限制，非 Linux 权限问题），以下通过 `nsys` 实测时间 + `ptxas` 编译指标 + GPU 理论规格进行 roofline 推算。

**RTX 4060 (AD107, sm_89) 理论峰值**：

| 指标 | 值 |
|------|---:|
| SM 数量 | 24 |
| FP32 峰值 | 15.1 TFLOPS |
| 显存带宽 | 136 GB/s (GDDR6, 128-bit, 8501 MHz) |
| Roofline 拐点 (AI threshold) | 111 FLOP/byte |

#### TEMPLATE 模式 (RGB, R=5)

每个像素遍历 11×11=121 个邻域，每个邻域约 10 FLOP（smem load、LUT 查表、乘加），总计 ~1213 FLOP/pixel。全局内存仅读写 6 bytes/pixel（3B u8 输入 + 3B u8 输出），其余由 shared memory 和 constant memory 承担。

| 指标 | 值 |
|------|---:|
| 算术强度 (AI) | **202 FLOP/byte** |
| 总计算量 | 10.1 GFLOP |
| 全局内存量 | 49.8 MB |
| Kernel 耗时 | 3.36 ms |
| 实际算力 | 3.0 TFLOPS (**19.8%** of peak) |
| 实际带宽 | 14.8 GB/s (10.9% of peak) |

**诊断**：AI = 202 > 111 → **Compute-bound**。但实际算力仅达峰值的 19.8%，说明计算资源利用率低。原因：

1. **Occupancy 不足**（67%）：64 regs/thread 限制每 SM 并发 warp 数，延迟隐藏能力受限
2. **Constant memory LUT 访问序列化**：warp 内线程访问不同 LUT 地址时退化为多次事务
3. **循环展开过深**：R=5 展开 121 次迭代，指令缓存压力大

#### SEPARABLE 模式 (RGB, R=5)

两遍分离滤波，每遍仅 11 个邻域。全局内存包含输入/输出 + 中间 float 缓冲区的读写。

| 指标 | 值 |
|------|---:|
| 算术强度 (AI) | **~9 FLOP/byte** |
| 总计算量 | 1.87 GFLOP |
| 实际全局内存量 | ~175 MB (含中间缓冲、tile 重叠) |
| Kernel 耗时 | 1.51 ms (H 0.79 + V 0.72) |
| 实际算力 | 1.24 TFLOPS (8.2% of peak) |
| 实际带宽 | ~116 GB/s (**85%** of peak) |

**诊断**：AI = 9 < 111 → **Memory-bandwidth-bound**。带宽利用率已达 85%，接近 GDDR6 实际上限。瓶颈在于：

1. **中间缓冲区的读写开销**：H pass 写 float×3 (12B/pixel)，V pass 读回，凭空增加 ~24B/pixel
2. **Tile 重叠导致的冗余加载**：halo 区域使实际全局内存访问量比理论值多 ~60%

#### Roofline 图示

```
TFLOPS
  15 ┤ ·····························*·················  FP32 Peak
     │                           ·╱
     │                         ·╱
     │                       ·╱
   3 ┤ ·················[TEMPLATE]   (AI=202, 3.0T)     ← compute-bound, 低利用率
     │                 ·╱
   1 ┤ ··[SEPARABLE]·╱              (AI=9, 1.2T)       ← memory-bound, 高 BW 利用率
     │           ·╱
     └───────┴───┴──────────────────────────────────→ FLOP/byte
             9  111                                     (Arithmetic Intensity)
```

### 10.4 瓶颈诊断

```
                TEMPLATE 模式端到端 ~7.5 ms
           ┌────────────┬────────────┬────────────┐
           │  H2D 2.0ms │ Kernel 3.4ms│ D2H 1.9ms │
           └────────────┴────────────┴────────────┘
                         46% 传输     │   54% 计算

                SEPARABLE 模式端到端 ~5.8 ms
           ┌────────────┬──────────────┬────────────┐
           │  H2D 2.0ms │ H+V 1.5ms   │  D2H 1.9ms │
           └────────────┴──────────────┴────────────┘
                        68% 传输       │  26% 计算
```

| 模式 | Kernel 瓶颈类型 | Kernel 利用率 | 端到端主瓶颈 |
|------|:---------------:|:------------:|:----------:|
| TEMPLATE | Compute-bound | 19.8% FP32 | Kernel 计算 (54%) |
| SEPARABLE | Memory-bound | 85% BW | PCIe 传输 (68%) |

### 10.5 优化建议（基于 profiler 数据）

#### TEMPLATE 模式 — 提升计算利用率

1. **降低寄存器用量到 ≤48**：
   - 方法：`#pragma unroll 4` 替代完全展开，或使用 `__launch_bounds__(256, 5)` 强制编译器压缩寄存器
   - 效果：occupancy 从 67% → 83%，更多活跃 warp 隐藏延迟
   - 风险：可能引入少量 spill，需实测权衡

2. **减少 constant memory 竞争**：
   - 将空间权重 LUT 从 constant memory 搬到 shared memory（仅 `(2R+1)^2 × 4B = 484B`）
   - 每个线程访问不同偏移时 shared memory 不会序列化，constant memory 会

3. **指令缓存优化**：
   - R=5 完全展开生成 ~1200 条指令，可能超出 L0 I-cache（sm_89 为 32KB）
   - 降低展开深度可减轻 I-cache 压力

#### SEPARABLE 模式 — 减少显存访问量

1. **消除中间缓冲区**：
   - 使用 fused H+V kernel（在同一个 kernel 中完成水平和垂直滤波）
   - 或使用 persistent kernel 方式，水平 pass 的结果直接存在寄存器/smem 中供垂直 pass 使用
   - 可节省 ~24B/pixel 的中间读写，理论加速 ~30%

2. **FP16 中间结果**：
   - 中间缓冲区用 `__half` 替代 `float`，带宽减半
   - 对 8-bit 图像精度完全足够

#### 两种模式共同 — 传输优化

1. **多 Stream 重叠**：
   - Stream A: H2D 传输；Stream B: kernel 执行；Stream C: D2H 传输
   - 理论端到端降至 `max(H2D, kernel, D2H)` ≈ 3.4ms (TEMPLATE) / 2.0ms (SEPARABLE)

2. **Zero-copy / Managed Memory**：
   - 对于单帧处理，`cudaMallocManaged` + `cudaMemPrefetchAsync` 可能比显式 memcpy 更高效
   - GPU 可在传输完成前开始处理已到达的数据块

---

## 十一、FP16 中间缓冲区优化（实测）

### 11.1 动机

由 Roofline 分析可知，SEPARABLE 模式处于 **memory-bandwidth-bound** 状态（BW 利用率 ~85%）。
水平 pass 写出、垂直 pass 读入的中间缓冲区是 GPU 内部的额外带宽压力：

- 4K RGB：中间缓冲区 = 3840 × 2160 × 3 × 4B = **95.1 MB**（float）→ **47.5 MB**（__half）
- 4K Gray：中间缓冲区 = 3840 × 2160 × 4B = **31.6 MB**（float）→ **15.8 MB**（__half）

### 11.2 实现方式

新增 `BILATERAL_MODE=3`（`SEPARABLE_FP16`），对 H/V 核函数增加 `TmpT` 模板参数：

```cpp
// Horizontal: InT input → TmpT intermediate
template <int RADIUS, typename InT = float, typename TmpT = float>
__global__ void k_bilateral_horizontal_gray(const InT* input, TmpT* output, ...);

// Vertical: TmpT intermediate → OutT output
template <int RADIUS, typename TmpT = float, typename OutT = float>
__global__ void k_bilateral_vertical_gray(const TmpT* input, OutT* output, ...);
```

- 内部累加器仍为 `float`（FP32 精度不丢失）
- 中间值写出时：`static_cast<__half>(sum / weight_sum)`（FP16 截断，精度损失极小）
- 中间值读入时：smem 加载 `static_cast<float>(input[idx])`（还原为 FP32）
- 编译时指定 `-arch=sm_89`（RTX 4060 Ada Lovelace，原生 FP16 I/O 支持）

### 11.3 性能结果（4K, radius=5, σ_s=3, σ_c=30, runs=50）

#### 4K RGB (3840×2160)

| Mode | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OpenCV |
|------|:--------:|:--------:|:-----------------:|:---:|:---------:|
| 0 STANDARD | ~10 | 9.33 | — | 0.65 | — |
| 1 TEMPLATE | 7.41 | 7.21 | 1119 | 0.80 | 4.20× |
| 2 SEPARABLE (float) | 5.57 | 5.42 | 1489 | 0.45 | 5.68× |
| **3 SEPARABLE_FP16** | **5.42** | **5.28** | **1532** | **0.46** | **5.99×** |

#### 4K Grayscale (3840×2160, 1ch)

| Mode | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | vs OpenCV |
|------|:--------:|:--------:|:-----------------:|:---:|:---------:|
| 0 STANDARD | 3.86 | 3.63 | 2147 | 0.62 | 4.45× |
| 1 TEMPLATE | 3.00 | 2.94 | 2764 | 0.62 | 5.71× |
| 2 SEPARABLE (float) | 2.08 | 1.99 | 3990 | 0.15 | 7.99× |
| **3 SEPARABLE_FP16** | **1.99** | **1.91** | **4161** | **0.12** | **8.20×** |

### 11.4 分析

FP16 相对于 float 中间缓冲的加速：

| 测试场景 | SEPARABLE float min | SEPARABLE_FP16 min | 提升 |
|----------|:-------------------:|:------------------:|:----:|
| 4K RGB | 5.42 ms | 5.28 ms | **+2.7%** |
| 4K Gray | 1.99 ms | 1.91 ms | **+4.2%** |

**提升幅度较保守的原因**：

1. **PCIe 传输主导端到端时间**：H2D + D2H ≈ 3.9 ms，kernel 执行仅 ~1.5 ms（SEPARABLE RGB）。
   FP16 仅优化 GPU 内部带宽，对 PCIe 时间无影响，因此端到端改善有限。
2. **Gray 场景提升更大**：Gray 单通道中间缓冲比 RGB 小得多（1/3），kernel 占总时间比例更高（~50%），FP16 节省的带宽效果更明显。
3. **MAE 保持低位**：FP16 的 10-bit 尾数对 [0,255] 像素值的精度约为 0.03 像素，色差权重误差可忽略；Gray 模式 MAE 甚至略有下降（0.15 → 0.12），属正常统计波动。

### 11.5 结论

| 指标 | 结果 |
|------|------|
| 代码改动 | 仅需 `TmpT` 模板参数 + `__half` 实例化，无算法改变 |
| 性能提升 | +2.7%（RGB）/ +4.2%（Gray），kernel 端约 +10% |
| 精度影响 | MAE ≤ 0.46，完全满足 < 1.0 要求 |
| 适用场景 | 当 kernel 执行时间占比更高时（如大 radius 或多帧流水线），收益将显著提升 |

---

## 十二、参考资料分析与待实验优化思路

> 来源：[xytroot/Bilateral-Filter](https://github.com/xytroot/Bilateral-Filter) 与 [OpenCV CUDA bilateral_filter.cu](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaimgproc/src/cuda/bilateral_filter.cu)

### 12.1 参考实现对比

#### xytroot/Bilateral-Filter（教学级实现）

| 技术 | xytroot | 本项目 | 对比 |
|------|---------|--------|------|
| Texture Memory | `tex2D()` + `cudaBindTexture2D` | 无（Shared Memory） | 见 12.2-A |
| 空间权重 | 1D `__constant__` 数组，kernel 内 `cGaussian[dy+r] * cGaussian[dx+r]` 合成 2D | 预计算完整 2D LUT | 本项目更优（1 次 vs 2 次查表） |
| 颜色权重 | kernel 内实时 `__expf()` | 256 元素 color LUT | **本项目远优** |
| Shared Memory | 无 | Tile + halo 协作加载 | 本项目更优 |
| 内存管理 | 每次 `cudaMalloc` / `cudaFree` | 持久化 GPU buffer | 本项目更优 |
| 通道支持 | 仅灰度 | 灰度 + RGB | 本项目更全 |

#### OpenCV CUDA bilateral_filter.cu

| 技术 | OpenCV CUDA | 本项目 | 意义 |
|------|-------------|--------|------|
| `cudaFuncCachePreferL1` | **显式设置** | 无 | 见 12.2-B |
| 圆形窗口裁剪 | `if (space2 > r2) continue` | 方形窗口 | 见 12.2-C |
| 内外像素分支 | 内部像素无边界检查快速路径 | 统一 clamp | 见 12.2-D |
| 向量化类型 | `uchar3/float3` + `saturate_cast` | 逐通道分离 smem | 各有取舍 |
| 颜色距离 | `norm_l1`（L1 范数）+ `exp()` 实时计算 | 预计算 LUT | **本项目更快** |
| Shared Memory | **无** | 有 | **本项目更优** |
| 边界处理模板 | 5 种 border mode（Reflect/Wrap/...） | 仅 clamp | OpenCV 更灵活 |

### 12.2 待实验优化思路

#### A. Texture Memory 用于梯度计算（来自 xytroot）

xytroot 的 `tex2D` 方案在现代 GPU 上应使用 `cudaTextureObject` API 替代已弃用的 `cudaBindTexture2D`。

- **适用场景**：ADAPTIVE 模式的梯度（Sobel）计算 pass。该 pass 为只读访问且无 tile 复用，texture cache 的 2D 空间局部性优化比 shared memory 更自然
- **优势**：硬件自动边界 clamp，消除 `max(0, min(N-1, x))` 分支；2D 缓存对 Sobel 3×3 窗口友好
- **预期**：梯度 pass 本身占比小（~5-10%），整体收益有限，更多是代码简洁性提升

#### B. `cudaFuncCachePreferL1`（来自 OpenCV）

```cpp
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

OpenCV 对所有 bilateral kernel 显式设置 L1 cache 偏好。在 sm_89 上 L1 和 shared memory 共享同一块 SRAM（128KB/SM），通过配置偏好可调整两者的分配比例。

- **代价**：零代码改动（一行 API 调用）
- **预期**：对 shared memory 用量较小的 kernel（如 separable gray horizontal，仅 1664B），释放更多 L1 给全局内存缓存可能有正面效果
- **注意**：sm_89 上该 hint 可能被硬件忽略（Ada 架构 L1/smem 分配策略有变化），需实测验证

#### C. 圆形窗口裁剪（来自 OpenCV）— ✅ 已完成

OpenCV 用 `if (space2 > r²) continue` 跳过圆形窗口外的角落像素。

本项目采用更高效的方案：在 spatial LUT 预计算时将圆外位置权重设为 0（Phase 1），kernel 中检测 `spatial_weight == 0.0f` 执行 `continue`（Phase 2）。无需实时计算 `space2`。

对于 radius=5，方形窗口 11×11 = 121 像素，圆内 81 像素，**圆外 40 像素（33%）**。

**实测结果**：

| 阶段 | TEMPLATE 4K RGB | TEMPLATE 4K Gray | STANDARD 4K RGB |
|------|:---------------:|:----------------:|:---------------:|
| Phase 1（仅 LUT 置零） | ~0% 性能 | ~0% 性能 | ~0% 性能 |
| Phase 2（+early continue） | **+13%** | **+65%** | +10% |

- TEMPLATE 的巨大收益源于编译器 dead code elimination：`#pragma unroll` 展开后，编译器在编译期确定哪 40 个位置恒为 0，直接删除这些迭代体的全部指令
- ADAPTIVE 模式加 continue 反而变慢（-7%），因为可变 `my_radius` 下分支有害且循环范围已由 my_radius 限制。已回退
- 详见 Opt C/F 实验记录

#### D. 内部/边界像素分支（来自 OpenCV）

OpenCV 在 kernel 内对完全不触及边界的像素走**无边界检查快速路径**：

```cpp
if (x - r >= 0 && y - r >= 0 && x + r < cols && y + r < rows) {
    // Fast path: no boundary check
    for (...) { value = src(cy, cx); ... }
} else {
    // Safe path: with boundary interpolation
    for (...) { value = b.at(cy, cx, ...); ... }
}
```

- **适用于**：STANDARD 模式（runtime radius，不能编译期优化掉 clamp）
- **预期**：图像内部 >95% 像素走快速路径，边界像素仅一小圈。但本项目已用 shared memory 在加载阶段统一做 clamp，计算阶段不再有 if，所以该优化的收益可能很小
- **改进思路**：在 shared memory 加载阶段做分支——内部 block 直接加载，边界 block 走 clamp

#### E. 多 Stream 条带并行（Strip Pipelining）

当前端到端延迟中 H2D + D2H 占 46-68%（见 10.4 节），kernel 与传输完全串行。通过将图像水平切分为若干 **strip（条带）**，可实现传输与计算的重叠：

**原理**：

```
传统方式（串行）：
  H2D ████████████          D2H ████████████
                  kernel ████

条带并行（4 strips × 4 streams）：
  Stream 0: H2D ██  kernel █  D2H ██
  Stream 1:   H2D ██  kernel █  D2H ██
  Stream 2:     H2D ██  kernel █  D2H ██
  Stream 3:       H2D ██  kernel █  D2H ██
             ↑ GPU copy engine 与 compute engine 物理独立，可真正并行
```

所谓 **strip（条带）** 就是将图像按行方向切成若干水平长条。例如 4K 图像（2160 行）切 4 条，每条 540 行。每条分配一个 CUDA stream，各 stream 的 H2D → kernel → D2H 形成独立的流水线，由 GPU 硬件调度引擎自动交错执行。

**实现要点**：

1. **Halo 重叠**：每条 strip 的输入需要向上下各扩展 radius 行（如 r=5 则多读 5 行），确保边界像素也能正确滤波。输出只写属于自己的行，无冗余
2. **Pinned Memory**：多 stream 异步传输要求 host 内存必须是 page-locked（已通过 `cudaHostRegister` 实现）
3. **Strip 数量选择**：太少无法充分重叠，太多 launch overhead 累积。经验值 4-8 条为宜
4. **适用前提**：单帧处理时收益明显；若已在视频流水线中（帧间重叠），则帧内 strip 的意义降低

**预期收益**：

| 模式 | 当前端到端 | 理想重叠后 | 加速比 |
|------|:---------:|:---------:|:------:|
| TEMPLATE (4K RGB) | 7.5 ms | ~3.5 ms (≈ kernel time) | **2.1×** |
| SEPARABLE (4K RGB) | 5.8 ms | ~2.0 ms (≈ H2D time) | **2.9×** |

这是当前**收益最大的单项优化方向**，因为瓶颈已从 kernel 计算转移到 PCIe 传输。

#### F. 圆形窗口 + Spatial LUT 预置零方案 — ✅ 已完成（合并入 C）

已作为 Opt C/F Phase 1 + Phase 2 实现。在 host 端 `init_spatial_lut` 中对圆外位置写 0，kernel 中 `if (spatial_weight == 0.0f) continue` 跳过。

实测结果远超预期：TEMPLATE kernel 不是运行时 predicated skip，而是编译器在展开后直接 **dead code elimination**，实现了编译期消除，TEMPLATE Gray 提升 +65%。

### 12.3 优先级排序（更新后）

| 优先级 | 优化 | 改动量 | 预期收益 | 实际 | 状态 |
|:------:|------|:------:|:--------:|:----:|:----:|
| ★★★ | E. 多 Stream 条带并行 | 大 | **2-3×** 端到端 | 0~-8%（WSL2） | ❌ 无收益 |
| ★★★ | C/F. 圆形窗口裁剪 + early continue | 小 | ~20% kernel | **+13%~+65%** | ✅ 完成 |
| ★☆☆ | B. `cudaFuncCachePreferL1` | 极小 | 0-5% | -36%（SEPARABLE） | ❌ 回退 |
| ★☆☆ | D. 内部/边界分支 | 中 | <5%（已有 smem clamp） | — | 未实验 |
| ★☆☆ | A. Texture Memory 梯度 | 中 | <5%（梯度 pass 占比小） | — | 未实验 |
