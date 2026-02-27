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
| Opt10 | **Block 尺寸 32×8**（Opt G 后） | 消除 warp 跨行导致的 smem bank conflict；32 线程在同一行，stride=1，零冲突 | `BLOCK_X=32, BLOCK_Y=8` (L29-33) |
| Opt11 | **SEPARABLE SoA 中间缓冲区**（Opt H） | 水平输出/垂直输入改为 R\|G\|B 平面格式，单通道 float 连续访问，合并效率从 ~33% 提升 | `k_bilateral_horizontal_rgb` 输出 SoA，`k_bilateral_vertical_rgb` 读取 SoA |
| Opt12 | **显式 fmaf（Opt I）** | 尝试将 `sum += n*w` 改为 `fmaf(n,w,sum)`，实测编译器已自动融合 | 所有 SEPARABLE kernel 中 `fmaf` 调用 |
| Opt13 | **SEPARABLE launch_bounds(256,6)（Opt K）** | 强制编译器将 regs 从 63 压到 40，occupancy 从 62%→97.5%，4K RGB -10.9% | `MIN_BLOCKS_PER_SM_SEP=6`，4 个 SEPARABLE kernel 的 `__launch_bounds__` |

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

> **注**：以下 Opt A~F 的实验数据来自 RTX 4060 (sm_89) 平台，为优化过程的历史记录。最终性能结果见第十章（Jetson AGX Thor 实测）。

测试环境（优化实验阶段）：RTX 4060 (sm_89, WSL2), 4K 图像（3840×2160），radius=5, σ_s=3, σ_c=30，50 次取均值。

### 基线性能（RTX 4060）

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

**结论**：性能退步 2.4x，MAE 也变差。原因：

1. halo 边界的 if/else 分支引入了 warp divergence
2. radius=5 时只有 10/32 个 lane 需要 halo，分支比例高
3. 原始 shared memory 方案加载已充分合并，延迟被计算隐藏
4. Warp shuffle 更适合纯寄存器场景（如 reduction），对这种需要大量邻域数据的滤波，shared memory 仍是更好的选择

### Opt C: SoA 数据布局（已回退）

将 RGB separable 改为 AoS->SoA 转换 + 3x 灰度 separable + SoA->AoS 转换。

| 指标 | 基线 | SoA 版 |
|------|:----:|:------:|
| 4K RGB SEPARABLE | 5.26 ms (min) | 7.15 ms (min) |
| MAE | 0.45 | 0.45 |

**结论**：性能退步 36%。原因：

1. AoS<->SoA 转换引入 2 次额外全局内存遍历（读+写各一次），4K RGB 约 47MB
2. 8 个 kernel launch 替代原来 2 个，launch overhead 累积明显
3. 每通道独立计算值域权重，丧失了 RGB kernel 中三通道共享权重的优势
4. 原始 RGB separable kernel 通过 shared memory 已经将 AoS 的不合并访问局限在加载阶段，计算阶段完全在 smem 中进行，瓶颈不在全局内存访问模式

### Opt D: `cudaFuncCachePreferL1` / `cudaFuncCachePreferShared`

在 sm_89 上 L1 cache 和 shared memory 共享 128 KB SRAM 池。测试了两种配置：

**L1 preference** (`cudaFuncCachePreferL1`):

| Mode | Before | After |
|------|:------:|:-----:|
| TEMPLATE 4K RGB min | 7.21 ms | 7.31 ms |
| SEPARABLE 4K RGB min | **5.42 ms** | **7.36 ms** (+36%) |

L1 偏好将 shared memory 从 64 KB 压缩到 32 KB/SM，导致 SEPARABLE 的 occupancy 下降，延迟隐藏能力降低。

**结论**：`cudaFuncCachePreferL1` 对本项目有害。改为 `PreferShared` 作为防御性注解。

---

### Opt E: Circular Window (Spatial LUT Corner Zeroing + Early Continue)

在 `init_spatial_lut` 中将 `dx^2 + dy^2 > radius^2` 的 LUT 项置零。
对于 r=5：121 个位置 -> 81 个在圆内，**40 个角落置零（33%）**。

#### Phase 1: LUT 预置零（无 kernel 分支）

| Mode | Old MAE | New MAE | Old PSNR | New PSNR |
|------|:-------:|:-------:|:--------:|:--------:|
| STANDARD RGB | 0.647 | **0.477** | 47.55 dB | **48.61 dB** |
| TEMPLATE RGB | 0.800 | **0.603** | 45.90 dB | **48.28 dB** |
| SEPARABLE RGB | 0.448 | **0.448** | 48.49 dB | 48.49 dB |
| ADAPTIVE RGB | 0.437 | **0.404** | 48.55 dB | **49.42 dB** |

Phase 1 结论：零性能成本，但精度显著改善（TEMPLATE RGB MAE 0.80->0.60，PSNR +2.4 dB）。

#### Phase 2: Early Continue（跳过圆外像素）

在 kernel 内循环中添加 `if (spatial_weight == 0.0f) continue;`，跳过圆外 33% 的迭代体。

| Mode | Before (ms) | After (ms) | 提升 | 说明 |
|------|:-----------:|:----------:|:----:|------|
| **TEMPLATE RGB** | 7.36 | **6.53** | **+13%** | 编译期常量 RADIUS -> 编译器 DCE 消除圆外迭代 |
| **TEMPLATE Gray** | 4.97 | **3.01** | **+65%** | 同上，Gray 内循环体更轻故比例更大 |
| STANDARD RGB | 9.44 | 8.61 | +10% | 运行时分支，warp 内一致（无 divergence） |
| ADAPTIVE RGB | 6.99 | 7.48 | **-7%** | 有害，已回退 |

**机制分析**：TEMPLATE kernel 的巨大收益源于编译器 dead code elimination——RADIUS 是编译期常量，`#pragma unroll` 完全展开 121 次迭代后，编译器在编译期确定哪 40 个位置的 `spatial_weight` 恒为 0，直接删除这些迭代体的全部指令。ADAPTIVE 加 continue 反而变慢，因为可变 `my_radius` 下循环范围已受限，额外分支只增加开销。

### Opt F: Multi-Stream Strip Pipelining（WSL2 无收益）

通过 `BILATERAL_STRIP=N` 将图像切分为 N 条水平 strip 到独立 CUDA stream。

| Mode | Single-shot | 2 strips | 4 strips | 8 strips |
|------|:-----------:|:--------:|:--------:|:--------:|
| TEMPLATE min | 7.16 ms | ~7.4 ms | 7.47 ms | 7.79 ms |
| SEPARABLE min | 5.39 ms | 5.51 ms | 5.68 ms | 5.84 ms |

**结论**：WSL2 虚拟化层序列化了 PCIe DMA 和 compute，无法实现真正的 copy+compute overlap。代码保留但默认关闭。

### Opt G: FP16 中间缓冲区（SEPARABLE_FP16）

新增 `BILATERAL_MODE=3`，SEPARABLE 水平/垂直 pass 之间的中间缓冲区从 float 改为 `__half`，带宽减半。

| 测试场景 | SEPARABLE float min | SEPARABLE_FP16 min | 提升 |
|----------|:-------------------:|:------------------:|:----:|
| 4K RGB (RTX 4060) | 5.42 ms | 5.28 ms | **+2.7%** |
| 4K Gray (RTX 4060) | 1.99 ms | 1.91 ms | **+4.2%** |

提升幅度有限的原因：PCIe 传输（H2D+D2H ~3.9ms）主导端到端时间，FP16 仅优化 GPU 内部带宽。

### Opt H: Block Size 32x8 — 消除 Shared Memory Bank Conflict（Thor 上验证）

**ncu 发现**：TEMPLATE kernel 在 16x16 block 下 shared memory load 存在 **2-way bank conflict**（50% excessive wavefronts），`short_scoreboard` stall 比率 3.92。

**根因分析**：16x16 block 中一个 warp（32 线程）跨两行（ty=0..1），stride=TILE_W_PAD=27。相邻行的 bank 偏移 = 27 mod 32 = 27，等价于 -5，导致前半 warp 和后半 warp 有 11/16 的 bank 重叠，产生 2-way conflict。

初始尝试 padding（TILE_W 从 26 改为 27）无效——50% conflict 不变。

**最终方案**：将 block 从 16x16 改为 **32x8**。每个 warp 的 32 个线程全部在同一行（tx=0..31），访问连续 32 个 float，stride=1，**零 bank conflict**。

**ncu 验证结果**：

| 指标 | 16x16 (优化前) | 32x8 (优化后) | 变化 |
|------|:-----------:|:----------:|:----:|
| Shared excessive wavefronts | 16,260,480 (50%) | **388,800 (2.3%)** | **-97.6%** |
| SM Throughput | 87.49% | **88.36%** | +0.87pp |
| Registers/thread | 23 → 28 | 28 | +5 (padding 的副作用) |

> 32x8 block 使 shared memory load 的 bank conflict 从 50% 降至 2.3%（剩余来自协作加载的 store 阶段）。throughput 提升幅度较小（+0.87pp），因为 TEMPLATE kernel 已处于高度优化状态（SM 88%），bank conflict 虽然减少但不再是唯一瓶颈。

**性能**：4K RGB benchmark min 持平（5.46ms → 5.46ms），Jellyfish 效应——bank conflict 减少被寄存器增加（23→28）部分抵消。

### Opt I: SEPARABLE 中间缓冲区 SoA 布局 — 改善 Global Memory Coalescing

**ncu 发现**：SEPARABLE vertical kernel 的全局内存合并效率仅 33%，68% 的 sector 传输是冗余的。原因是中间缓冲区为 AoS (RGB 交错) float 格式。

**方案**：水平 kernel 输出改为 SoA 布局（R 平面 | G 平面 | B 平面），垂直 kernel 从 SoA 读取。垂直 kernel 读取单通道时每个 warp 访问连续 float 地址，完全合并。

**ncu 验证结果**：

| 指标 | AoS (优化前) | SoA (优化后) | 变化 |
|------|:-----------:|:----------:|:----:|
| H: global uncoalesced | 69% | **47%** | **-22pp** |
| V: global uncoalesced | 68% | **29%** | **-39pp** |

> 垂直 kernel 的全局内存冗余从 68% 降至 29%（中间缓冲区读取完全合并），水平 kernel 也有 22pp 改善（输出写入变为合并）。输入/输出仍为 u8 RGB AoS（无法改变外部格式），是剩余 uncoalesced 的来源。

**性能**：4K RGB SEPARABLE min 持平（3.39ms），1080p min=0.81ms，gray min=0.36ms。

### Opt J: FMA 融合（fmaf）— 编译器已自动优化

**ncu 发现**：37% 非融合 FP32 指令，预估 +9.5% 收益。

**方案**：将 `sum += neighbor * w` 改为 `sum = fmaf(neighbor, w, sum)` 显式 FMA。

**结果**：ncu 显示 fused/non-fused 比率**未变化**（~0.36）。nvcc `-O3` 已自动将所有可融合的 MUL+ADD 对生成 FFMA 指令。剩余的 "non-fused" 是独立的 FADD（如 `weight_sum += w`）和 FMUL（如 `spatial_weight * color_weight`），没有配对的加法/乘法可融合。

> **教训**：nvcc `-O3` 的 FMA 融合已接近最优。`fmaf` 显式声明不会超越编译器自动优化，仅作为代码可读性标注保留。

### Opt K: SEPARABLE 激进 Launch Bounds — 降低寄存器提升 Occupancy

**ncu 发现**：SEPARABLE RGB kernels 使用 61-63 regs/thread，occupancy 仅 62-63%（受 regs 限制为 4 blocks/SM）。ncu 预估提升 occupancy 可获 10-20% 收益。

**方案**：为 SEPARABLE kernels 单独设置 `__launch_bounds__(256, 6)`（`MIN_BLOCKS_PER_SM_SEP=6`），强制编译器将寄存器压到 ≤42（65536 / 256 / 6 = 42.67）。

**ptxas 结果**：

| Kernel | 优化前 Regs | 优化后 Regs | Spill | Occupancy (理论) |
|--------|:-----------:|:-----------:|:-----:|:----------------:|
| `horizontal_rgb<5>` | 63 | **40** | **0** | **100%** |
| `vertical_rgb<5>` | 62 | **40** | **0** | **100%** |
| `horizontal_gray<5>` | 33 | 33 | 0 | 不变 |
| `vertical_gray<5>` | 33 | 33 | 0 | 不变 |

**ncu 验证**：

| 指标 | 优化前 | 优化后 | 变化 |
|------|:------:|:------:|:----:|
| Achieved Occupancy (H) | 62.1% | **97.7%** | **+35.6pp** |
| Achieved Occupancy (V) | 63.1% | **97.5%** | **+34.4pp** |
| SM Throughput (H) | 64.5% | **79.4%** | **+14.9pp** |
| SM Throughput (V) | 65.0% | **73.2%** | **+8.2pp** |
| Not-selected stall (H) | — | 34% (4.8 cycles) | 高 occupancy 带来 warp 调度竞争 |

**性能**：

| 测试 | 优化前 min | 优化后 min | 变化 |
|------|:--------:|:--------:|:----:|
| 4K RGB | 3.39 ms | **3.02 ms** | **-10.9%** |
| 1080p RGB | 0.81 ms | **0.77 ms** | **-4.9%** |
| 4K Gray | 1.40 ms | 1.40 ms | ~持平（Gray 已是 33 regs） |

> **关键洞察**：这是单一改动（一个宏常量）带来最大端到端收益的优化。sm_110 编译器有足够能力在不 spill 的前提下将 63 regs 压缩到 40——这证明之前的寄存器分配过于保守。`__launch_bounds__` 的 `minBlocksPerSM` 参数是影响 CUDA 编译器寄存器分配策略的最有效杠杆。

### Opt L: u8 RGB 向量化加载 — ncu 分析后取消实施

**ncu 发现**：水平 RGB kernel 的 global load sectors/request = 2.98（理论最优 4.0），利用率 75%，符合 RGB 3B/像素的理论极限（3/4 = 75%）。

**分析**：RGB u8 格式每像素 3 字节，不是 4/8/16 字节对齐。相邻线程地址间距 3B，导致 128B sector 利用率天然为 ~75%。除非将输入格式改为 RGBX（4B 对齐），否则无法提升。而改变格式需要修改外部接口，代价过高。

**结论**：sectors/request = 2.98 已是 RGB u8 AoS 格式的理论极限，**不值得实施**。同时 SM throughput 已达 75-81%，优化 load 效率对端到端影响有限。

### Opt M: Fused H+V 单 Kernel — 实验失败

**动机**：消除 SEPARABLE 两趟之间的中间 float buffer 读写（4K RGB: 2 × 8.3M × 3ch × 4B = 192 MB 全局带宽），同时减少 1 次 kernel launch。

**方案**：单 kernel 内三阶段——Phase 0: 加载 2D halo (FUSED_H × TILE_W) → Phase 1: 对所有 FUSED_H 行做水平滤波 → Phase 2: 从水平结果做垂直滤波输出。

**smem 用量**（RGB, R=5, 32x8 block）：
- smem_raw: 3 × 18 × 43 × 4B = 9.3 KB（原始数据 + 2D halo）
- smem_h: 3 × 18 × 32 × 4B = 6.9 KB（水平滤波结果）
- 总计：~16.2 KB/block

**实测结果**：

| 测试 | SEPARABLE (ms) | FUSED (ms) | 变化 |
|------|:------------:|:--------:|:----:|
| 4K RGB | 3.02 | **3.96** | **-31%** |
| 4K Gray | 1.40 | **1.60** | **-14%** |
| 1080p RGB | 0.77 | **1.03** | **-34%** |

**性能倒退原因**：
1. **Phase 1 计算膨胀**：每 block 需处理 `FUSED_H × BLOCK_X = 18 × 32 = 576` 个水平滤波点，256 线程每个做 2-3 次水平循环（11 次迭代/次），计算量远超 SEPARABLE 的 1 次/线程
2. **smem 容量翻倍**：6 个 smem 数组（vs SEPARABLE 每 kernel 3 个），可能触发 register spill 或降低可同时执行的 block 数
3. **多一次 barrier**：3 阶段需 2 次 `__syncthreads()`（vs SEPARABLE 每 kernel 1 次）
4. **关键**：Thor 统一内存的全局带宽 ~120 GB/s，中间 buffer 读写仅占总时间 ~10-15%。节省的带宽远不足以补偿增加的计算开销

> **教训**：Fused kernel 仅在中间缓冲区成本占比很高（如 PCIe 独显环境下 > 30%）时才有价值。Thor 的统一内存使中间 buffer 开销很小，不值得为此增加 block 内计算复杂度。

### Opt N: FP16 中间缓冲区（Thor 平台验证） — 小幅有效

**方案**：SEPARABLE 两趟之间的中间缓冲区从 `float`（4B）改为 `__half`（2B），带宽减半。利用现有模板参数 `TmpT` 直接实例化 `<5, uint8_t, __half>` 和 `<5, __half, uint8_t>`，计算仍在 FP32 进行，精度无损。

**ncu 验证**（1080p RGB, vs FP32 intermediate）：

| 指标 | FP32 Intermediate | FP16 Intermediate | 变化 |
|------|:-----------------:|:-----------------:|:----:|
| V kernel LD sectors/req | 4.00 | **2.00** | **-50%** |
| H kernel ST sectors/req | 4.00 | **2.00** | **-50%** |
| V kernel L1 LD hit rate | ~4% | ~4% | 不变（列访问局部性差） |
| FP16 pipe utilization | 0% | **~2%** | 仅 half↔float 转换 |

**实测结果**：

| 测试 | SEPARABLE (ms) | SEP_FP16 (ms) | 变化 |
|------|:--------------:|:-------------:|:----:|
| 4K RGB | 3.02 | **2.97** | **-1.7%** |
| 4K Gray | 1.40 | **1.30** | **-7.1%** |
| 1080p RGB | 0.77 | **0.77** | ~持平 |
| 1080p Gray | 0.40 | **0.36** | **-10%** |

> **分析**：FP16 中间缓冲区在 Gray 模式下收益更大（-7%~-10%），因为 Gray 的中间 buffer 占比更高。RGB 收益有限（-1.7%），因为中间 buffer 已改为 SoA 布局（Opt H），合并效率已经很好。FP16 进一步减半了每次全局事务的字节数，但 V kernel 的 L1 命中率仍为 ~4%（列方向局部性天然差），限制了收益。

### Opt N2: FP16 全量计算 — 实验失败

**动机**：Thor (Blackwell) 的 FP16 吞吐量理论上是 FP32 的 2 倍。将 smem 和内循环累加全部改为 `__half`，期望利用 2x FP16 吞吐。

**方案**：新增专用 FP16 compute kernel（`k_bilateral_horizontal_rgb_fp16` 等），smem 用 `__half`（减半带宽），`__hfma` / `__hadd` / `__hdiv` 全用 FP16 指令，LUT 查找后 `__float2half()` 转换。

**实测结果**：

| 测试 | SEP_FP16 Intermediate (ms) | FP16 Full Compute (ms) | 变化 |
|------|:--------------------------:|:----------------------:|:----:|
| 4K RGB | 2.97 | **3.22** | **-8.4%（倒退）** |

**精度**：MAE 0.68（vs FP32 的 0.45），PSNR 47.36 dB（vs 48.49），仍通过 < 1.0 / > 40 dB 阈值。

**性能倒退原因**：
1. **FP16 标量无吞吐优势**：sm_110 的 2x FP16 优势仅在 `__half2` 打包操作（如 `__hfma2`）上。单个 `__hfma` 与 FP32 `fmaf` 吞吐相同（均 1 cycle）
2. **float↔half 转换开销**：每次循环迭代需 6+ 次 `__float2half()` / `__half2float()`（LUT 返回 float，颜色差计算需 float 精度），转换指令抵消了 smem 带宽节省
3. **`__hdiv` 慢**：FP16 除法无硬件指令，编译器用 reciprocal+multiply 模拟

> **教训**：FP16 优化仅在以下条件同时满足时有效：(a) 使用 `__half2` 打包操作获得 2x 吞吐，(b) 数据源和目标都是 FP16（避免转换），(c) 无 FP32 LUT 依赖。当前 bilateral filter 的 constant memory LUT 是 float 类型，使条件 (b)(c) 无法满足。

### 优化实验总结

| 优化 | 状态 | 实际效果 | 教训 |
|------|:----:|---------|------|
| `__launch_bounds__` | 保留 | ~持平 | 寄存器压力低时收益有限，但无副作用 |
| Warp Shuffle | 回退 | -2.4x | 需要 halo 的滤波场景不适合 shuffle |
| SoA 布局 (host 端) | 回退 | -36% | 格式转换开销 > 合并访问收益 |
| `cudaFuncCachePreferL1` | 回退 | -36% | L1 偏好压缩 smem 容量，降低 occupancy |
| 圆形窗口 LUT 预置零 | 保留 | MAE 改善 0.15~0.20 | 零性能成本的精度优化 |
| 圆形窗口 early-continue | TEMPLATE/STANDARD 启用 | **+13%~+65%** | 编译期常量使编译器彻底消除圆外迭代 |
| Strip Pipeline | 无收益（代码保留） | 0~-8% | WSL2 阻止 copy+compute 并行 |
| FP16 中间缓冲 (RTX 4060) | 保留 | +2.7%~+4.2% | PCIe 传输主导时收益有限 |
| **Block 32x8** | **保留** | **smem conflict -97.6%** | warp 跨行是 bank conflict 根因，32x8 确保 warp 全在一行 |
| **SEPARABLE SoA 中间缓冲** | **保留** | **uncoalesced -22~39pp** | 中间缓冲区 SoA 布局让垂直 pass 完全合并 |
| **fmaf 显式 FMA** | 保留（无效果） | 0% | nvcc -O3 已自动最优融合 |
| **SEPARABLE launch_bounds(256,6)** | **保留** | **4K RGB -10.9%, SM +14pp** | regs 63→40, occupancy 62%→97.5%, 零 spill |
| **FP16 中间缓冲 (Thor)** | **保留 (MODE=3)** | **Gray -7~10%, RGB -1.7%** | 中间 buffer 带宽减半，Gray 收益显著 |
| u8 RGB 向量化 | 取消实施 | sectors/req=2.98≈极限 | RGB 3B/像素是 AoS 格式天然限制 |
| Fused H+V Kernel | 实验失败 | **-31~34%** | 统一内存下中间 buffer 成本低，不值得增加 block 计算复杂度 |
| FP16 全量计算 | 实验失败 | **-8.4%** | 标量 __half 无 2x 优势，float↔half 转换开销高 |

> **启示**：ncu 数据驱动的精细调优成效显著。Opt K（激进 launch_bounds）是最成功的单一优化：仅改一个常量（`MIN_BLOCKS_PER_SM_SEP=6`）即让编译器将寄存器从 63 降到 40，occupancy 从 62% 跃升到 97.5%，4K RGB 实测提速 10.9%。关键是**零 spill**——sm_110 编译器有足够能力在不溢出的前提下压缩寄存器。Opt N（FP16 中间缓冲区）是本轮唯一成功的新优化，在 Gray 模式下取得了显著收益。三个"失败"实验（Opt L/M/N2）同样有价值——它们通过实测验证了优化边界，避免了在无效方向上的继续投入。

---

## 十、Jetson AGX Thor 平台实测

### 10.1 测试环境

| 项目 | 值 |
|------|---:|
| **平台** | NVIDIA Jetson AGX Thor Developer Kit |
| **GPU** | NVIDIA Thor (Blackwell, sm_110 / compute_11.0) |
| **SM 数量** | 20 |
| **统一内存** | 128 GB LPDDR5x（CPU/GPU 共享） |
| **L2 Cache** | 32 MB |
| **Shared Memory/SM** | 228 KB |
| **寄存器/SM** | 65536 |
| **最大线程/SM** | 1536 |
| **CUDA** | 13.0 |
| **Driver** | 580.00 |
| **JetPack** | R38.2.1 |
| **OpenCV** | 4.x（CPU only，无 CUDA 模块） |

**关键架构特性**：Jetson AGX Thor 采用**统一内存架构（Unified Memory）**，CPU 和 GPU 共享物理内存，无 PCIe 总线。`cudaMemcpy` 本质上是内存拷贝而非 DMA 传输，开销远小于独显平台。这从根本上改变了性能瓶颈分布。

### 10.2 滤波参数

```
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

Benchmark 方法：5 次 warmup + 50 次计时，报告 mean +/- stddev。由于 Jetson 平台存在 DVFS（动态频率调节），数据波动较独显平台大，**min 值更能反映 GPU 稳定性能**。

### 10.3 4K RGB 性能结果 (3840x2160x3)

> 数据版本：Opt G/H/I/K/N 之后（32x8 block，SoA 中间缓冲区，SEPARABLE launch_bounds(256,6)，FP16 中间缓冲区），5 warmup + 50 runs

| Implementation | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | PSNR (dB) | vs OCV CPU |
|----------------|:--------:|:--------:|:-----------------:|:---:|:---------:|:----------:|
| **SEP_FP16** | **3.03** | **2.97** | **2741** | **0.46** | **48.39** | **28.0x** |
| SEPARABLE | 3.10 | 3.02 | 2673 | 0.45 | 48.49 | 27.2x |
| FUSED | 3.98 | 3.96 | 2083 | 0.45 | 48.49 | 21.1x |
| TEMPLATE | 5.50 | 5.47 | 1508 | 0.60 | 48.28 | 15.3x |
| ADAPTIVE | 6.16 | 6.13 | 1346 | 0.40 | 49.42 | 13.7x |
| STANDARD | 9.30 | 9.28 | 892 | 0.48 | 48.61 | 9.1x |
| OpenCV CPU | ~84 | ~83 | ~99 | — | — | 1.0x |

### 10.4 4K Grayscale 性能结果 (3840x2160x1)

| Implementation | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | PSNR (dB) | vs OCV CPU |
|----------------|:--------:|:--------:|:-----------------:|:---:|:---------:|:----------:|
| **SEP_FP16** | **1.40** | **1.30** | **5915** | **0.12** | **57.00** | **39.2x** |
| SEPARABLE | 1.42 | 1.40 | 5849 | 0.15 | 56.18 | 37.3x |
| FUSED | 1.72 | 1.60 | 4809 | 0.15 | 56.18 | 30.9x |
| TEMPLATE | 3.53 | 3.46 | 2348 | 0.61 | 50.23 | 15.0x |
| STANDARD | 4.35 | 4.33 | 1906 | 0.61 | 50.23 | 12.2x |
| ADAPTIVE | 5.82 | 5.78 | 1426 | 0.61 | 50.23 | 9.2x |
| OpenCV CPU | ~53 | ~52 | ~157 | — | — | 1.0x |

### 10.5 1080p 性能结果 (1920x1080)

#### 1080p RGB

| Implementation | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | PSNR (dB) | vs OCV CPU |
|----------------|:--------:|:--------:|:-----------------:|:---:|:---------:|:----------:|
| **SEP_FP16** | **0.88** | **0.77** | **2351** | **0.46** | **48.36** | **31.4x** |
| SEPARABLE | 0.85 | 0.77 | 2434 | 0.45 | 48.46 | 32.6x |
| FUSED | 1.14 | 1.03 | 1821 | 0.45 | 48.46 | 24.6x |
| TEMPLATE | 1.54 | 1.41 | 1344 | 0.61 | 48.21 | 17.9x |
| ADAPTIVE | 1.71 | 1.58 | 1211 | 0.41 | 49.34 | 16.2x |
| STANDARD | 2.40 | 2.37 | 862 | 0.48 | 48.58 | 11.5x |
| OpenCV CPU | ~28 | ~27 | ~75 | — | — | 1.0x |

#### 1080p Grayscale

| Implementation | Avg (ms) | Min (ms) | Throughput (MP/s) | MAE | PSNR (dB) | vs OCV CPU |
|----------------|:--------:|:--------:|:-----------------:|:---:|:---------:|:----------:|
| **SEP_FP16** | **0.40** | **0.36** | **5139** | **0.12** | **56.85** | **38.8x** |
| SEPARABLE | 0.46 | 0.40 | 4521 | 0.15 | 56.06 | 34.2x |
| FUSED | 0.50 | 0.43 | 4110 | 0.15 | 56.06 | 30.9x |
| TEMPLATE | 0.97 | 0.90 | 2129 | 0.61 | 50.18 | 16.0x |
| STANDARD | 1.18 | 1.12 | 1752 | 0.61 | 50.18 | 13.2x |
| ADAPTIVE | 1.61 | 1.48 | 1290 | 0.61 | 50.18 | 9.7x |
| OpenCV CPU | ~16 | ~14 | ~130 | — | — | 1.0x |

### 10.6 质量验证

所有模式和测试场景均通过验证：

| 模式 | MAE (RGB) | MAE (Gray) | PSNR (RGB) | PSNR (Gray) | 状态 |
|------|:---------:|:----------:|:----------:|:-----------:|:----:|
| STANDARD | 0.48 | 0.61 | 48.61 dB | 50.23 dB | < 1.0 / > 40 dB |
| TEMPLATE | 0.60 | 0.61 | 48.28 dB | 50.23 dB | < 1.0 / > 40 dB |
| SEPARABLE | 0.45 | 0.15 | 48.49 dB | 56.18 dB | < 1.0 / > 40 dB |
| SEP_FP16 | 0.46 | 0.12 | 48.39 dB | 57.00 dB | < 1.0 / > 40 dB |
| ADAPTIVE | 0.40 | 0.61 | 49.42 dB | 50.23 dB | < 1.0 / > 40 dB |
| FUSED | 0.45 | 0.15 | 48.49 dB | 56.18 dB | < 1.0 / > 40 dB |

---

## 十一、Profiler 分析（Jetson AGX Thor）

**测试环境**：Jetson AGX Thor (sm_110, Blackwell), 20 SM, 128 GB 统一内存, SM 频率 1.572 GHz
**测试数据**：4K RGB (3840x2160x3), radius=5, sigma_s=3, sigma_c=30
**工具**：`ncu` (Nsight Compute 2025.3.1) `--set full` 采集硬件计数器 + `nsys` (Nsight Systems 2025.3.2) 采集时间线

> 注：Thor 为原生 Linux 环境（非 WSL2），`perf_event_paranoid=1`，ncu 可完整采集 GPU 硬件计数器。

### 11.1 nsys 时间线分析

#### TEMPLATE 模式 (MODE=1, 55 runs)

| 阶段 | 耗时 | 占比 | 说明 |
|------|-----:|-----:|------|
| `k_bilateral_filter_rgb_template<5,u8,u8>` | 4.98 ms/次 | **91%** | 唯一的 GPU kernel |
| `cudaMemcpy H2D` (24.9 MB) | 0.20 ms/次 | 3.6% | 统一内存拷贝 |
| `cudaMemcpy D2H` (24.9 MB) | 0.21 ms/次 | 3.8% | 统一内存拷贝 |
| `cudaDeviceSynchronize` | 5.01 ms/次 | — | 等待 kernel 完成 |
| `cudaLaunchKernel` | 0.008 ms | <0.1% | 可忽略 |

**传输带宽**：H2D 总 1369 MB / 11.25 ms = **122 GB/s**，D2H 总 1369 MB / 11.74 ms = **117 GB/s**。统一内存直接走片上互联，远高于 RTX 4060 的 PCIe 4.0（12 GB/s）。

#### SEPARABLE 模式 (MODE=2, 55 runs)

| 阶段 | 耗时 | 占比 | 说明 |
|------|-----:|-----:|------|
| `k_bilateral_horizontal_rgb<5,u8,float>` | 1.52 ms/次 | **50.7%** | 水平 pass |
| `k_bilateral_vertical_rgb<5,float,u8>` | 1.48 ms/次 | **49.3%** | 垂直 pass |
| `cudaMemcpy` (H2D+D2H 合计) | 0.42 ms/次 | ~12% | 含中间缓冲区分配 |
| `cudaLaunchKernel` | 0.008 ms/次 | <0.1% | 2 次 launch |

**关键发现**：

1. **统一内存消除了 PCIe 传输瓶颈**：cudaMemcpy 仅占端到端时间 ~5%（RTX 4060 上占 46-68%）。
2. **Kernel 执行主导端到端时间**：TEMPLATE 91%，SEPARABLE 的 H+V 合计 3.0 ms 占 ~88%。
3. **两个 separable pass 几乎等分**：水平 1.52 ms vs 垂直 1.48 ms。

### 11.2 ncu 硬件计数器分析

#### 11.2.1 TEMPLATE 模式 — `k_bilateral_filter_rgb_template<5,u8,u8>`

**Launch Configuration** (Opt G 后):
- Block: (32, 8, 1) = 256 threads，Grid: (120, 270, 1) = 32,400 blocks
- Waves per SM: 270，SM 频率: 1.572 GHz

##### Speed-of-Light (SOL) 概览

| 指标 | 值 | 说明 |
|------|---:|------|
| **SM Throughput** | **88.83%** | 接近计算峰值 |
| **L1/TEX Throughput** | **87.28%** | Shared memory 访问密集 |
| **L2 Throughput** | 3.50% | 数据集 fit in L2 (32 MB) |
| **DRAM Throughput** | ~0% | 统一内存数据由 L2/sysmem 路径服务 |
| **Duration** | 4.975 ms | nsys 一致 |
| **瓶颈诊断** | **Compute-bound** | SM 88.8% >> DRAM ~0% |

##### Occupancy

| 指标 | 值 |
|------|---:|
| **理论 Occupancy** | 100% (48/48 warps) |
| **实测 Occupancy** | **97.76%** (46.93/48 warps) |
| 寄存器/线程 | 23（分配 24） |
| Occupancy 限制因素 | 寄存器（10 blocks/SM） |

> sm_89 (RTX 4060) 上同一 kernel 使用 64 regs，occupancy 仅 67%。Blackwell 编译器的寄存器优化带来了 **+31% occupancy 提升**。

##### 指令执行效率

| 指标 | 值 |
|------|---:|
| IPC (Instructions Per Cycle, active) | 3.55 inst/cycle |
| IPC (elapsed) | 71.06 inst/cycle (20 SMs 合计) |
| Issue Slots Busy | 88.83% |
| 总执行指令数 | 555,692,400 |

##### 流水线利用率

| 流水线 | 利用率 | 说明 |
|--------|-------:|------|
| **LSU** (Load/Store) | **43.84%** | Shared memory 读写密集 |
| **FMA** (浮点乘加) | **40.99%** | 权重计算、加权求和 |
| **ADU** (地址发散) | 38.77% | smem 地址计算 |
| **ALU** (整数) | 30.41% | 索引计算 |
| **XU** (超越函数) | 28.17% | 注：LUT 查表仍有部分 exp 路径 |
| FP16 | 6.79% | 少量类型转换 |

> LSU 与 FMA 流水线**负载均衡**（43.8% vs 41.0%），说明计算与访存交织良好。

##### 缓存命中率

| 缓存层 | 命中率 | 说明 |
|--------|-------:|------|
| **Constant Cache** (LUT) | **99.99%** | spatial_lut + color_lut 完美缓存 |
| **Instruction Cache** | **100.0%** | 无 I-cache miss |
| **L1/TEX** | **68.53%** | 全局内存 load 75.3%，store 48.2% |
| **L2** | **73.31%** | read 66.2%，write 81.9% |

##### Shared Memory Bank Conflicts

| 指标 | 16x16 (Opt G 前) | 32x8 (Opt G 后) | 变化 |
|------|:-----------------:|:----------------:|:----:|
| Shared excessive wavefronts | 16,260,480 (50%) | **388,800 (2.3%)** | **-97.6%** |
| Shared store conflict | 1.2-way (13%) | 1.9-way (47%) | store 略增 |
| ncu Est. Speedup (shared) | 49.66% | **2.32%** | 已非瓶颈 |

**分析**：Opt G 前（16x16 block），shared memory load 存在 2-way bank conflict（50%），原因是 warp 跨两行访问 smem，行间 stride=27 mod 32=27，导致前半和后半 warp 有 11 个 bank 重叠。改为 32x8 block 后 warp 全在一行内（stride=1），**load conflict 降至 2.3%**。剩余的 excessive 来自协作加载阶段的 store（1.9-way），但 store 只执行一次而 load 在 81 次迭代中每次都执行，影响极小。

##### Warp Stall 分析

| Stall 原因 | 比率 | PC 采样数 | 说明 |
|-----------|-----:|----------:|------|
| **short_scoreboard** | **3.92** | 45,780 | Shared memory / L1 依赖等待 |
| **not_selected** | 3.21 | 36,679 | 有资格但未被调度 |
| **wait** | 2.79 | 32,471 | 固定延迟依赖 |
| long_scoreboard | 0.66 | 7,188 | L2/DRAM 依赖 |
| no_instruction | 0.42 | 5,101 | 指令缓存 |
| math_pipe_throttle | 0.40 | 4,654 | 计算流水线满 |
| barrier | 0.33 | 3,340 | `__syncthreads()` |
| dispatch_stall | 0.37 | 4,334 | 调度器延迟 |

**诊断**：Opt G 前，**short_scoreboard 是最大 stall 源**（比率 3.92，占采样 30%），与 bank conflict 1.97x 一致。Opt G（32x8 block）消除 97.6% 的 bank conflict 后，short_scoreboard stall 预计大幅下降，SM throughput 从 87.49% 提升至 88.36%。

#### 11.2.2 SEPARABLE 模式 — Horizontal + Vertical

##### Speed-of-Light (SOL) 概览

| 指标 | Horizontal | Vertical | 说明 |
|------|:----------:|:--------:|------|
| **SM Throughput** | **64.87%** | **67.28%** | 中等计算利用率 |
| **Memory Throughput** | **57.09%** | **34.16%** | H 更大（输入 u8 不合并） |
| **L1/TEX Throughput** | 56.39% | 34.23% | |
| **L2 Throughput** | 34.44% | 18.20% | |
| **Duration** | 1.50 ms | 1.46 ms | |
| **瓶颈诊断** | **Compute+Memory 均衡** | **Compute-bound** | |

##### Occupancy

| 指标 | H (Opt K 前→后) | V (Opt K 前→后) |
|------|:----------:|:--------:|
| 寄存器/线程 | 63 → **40** | 62 → **40** |
| 理论 Occupancy | 66.67% → **100%** | 66.67% → **100%** |
| **实测 Occupancy** | 62.09% → **97.67%** | 63.10% → **97.49%** |
| Occupancy 限制因素 | 寄存器 (4→6 blocks/SM) | 寄存器 (4→6 blocks/SM) |

> Opt K 通过 `__launch_bounds__(256, 6)` 将寄存器从 61-63 压缩到 40（零 spill），occupancy 从 62% 跃升到 97.5%。SM throughput 从 65% 提升至 73-79%，4K RGB 端到端提速 10.9%。

##### 全局内存合并效率

| 指标 | Horizontal (AoS→SoA) | Vertical (SoA→AoS) | 变化 |
|------|:----------:|:--------:|:----:|
| Global uncoalesced (Opt H 前) | **69%** | **68%** | — |
| Global uncoalesced (Opt H 后) | **47%** | **29%** | **-22pp / -39pp** |
| Global Load 利用率 | 7.1/32 bytes/sector (22%) | 7.1/32 (22%) | — |
| Global Store 利用率 | 10.7/32 (33%) | 8.0/32 (25%) | — |

**分析**：Opt H（SoA 中间缓冲区）大幅改善了垂直 kernel 的合并效率——uncoalesced 从 68% 降至 29%（-39pp），因为垂直 pass 从 SoA 平面读取单通道 float，完全合并。水平 kernel 也有 22pp 改善（SoA 写入合并）。

剩余的 uncoalesced 来自**外部格式限制**：输入/输出为 u8 RGB AoS（3 bytes/pixel），无法对齐 32-byte sector。彻底解决需要改变外部 I/O 格式（uchar4 padding 或 SoA 全链路），但这超出滤波 kernel 的范围。

##### Shared Memory Bank Conflicts

| 指标 | Horizontal | Vertical |
|------|:----------:|:--------:|
| Load bank conflict (n-way) | **2.1-way** | — |
| Store bank conflict (n-way) | **2.1-way** | 1.2-way |
| ncu 预估加速（消除 conflict） | Load +29%, Store +30% | Store +5% |

##### FP32 利用率

| 指标 | Horizontal | Vertical |
|------|:----------:|:--------:|
| FP32 峰值利用率 | **16%** | **16%** |
| 非融合 FP32 指令 | 34,214,400 | 34,214,400 |
| 可 FMA 融合指令 | 12,441,600 | 12,441,600 |
| FMA 融合后预估加速 | **+37%** | **+37%** |

> 大量 FP32 加法和乘法指令未被编译器融合为 FMA，这是 sm_110 编译器的优化盲区。手动使用 `__fmaf_rn` 或重排表达式可促进融合。

##### 分支效率

| 指标 | Horizontal | Vertical |
|------|:----------:|:--------:|
| 分支效率 | **83.38%** | **100%** |
| Divergent branches | 3,267 | 0 |

Horizontal 的分支发散来自 halo 区域加载的边界检查。

### 11.3 编译期指标（ptxas, sm_110 vs sm_89）

#### R=5 u8 kernel 资源使用对比

| Kernel | sm_110 Regs | sm_89 Regs | sm_110 Smem | Spill | 变化 |
|--------|:-----------:|:----------:|:-----------:|:-----:|:----:|
| `rgb_template<5,u8,u8>` | **28** | 64 | 9296 B | 0 | **-56%** (Opt G: 32x8 block) |
| `gray_template<5,u8,u8>` | **21** | 63 | 3104 B | 0 | **-67%** |
| `horizontal_rgb<5,u8>` | **40** | 62 | 4128 B | 0 | **-35%** (Opt K: launch_bounds(256,6)) |
| `vertical_rgb<5,u8>` | **40** | 62 | 6912 B | 0 | **-35%** (Opt K) |
| `horizontal_gray<5,u8>` | 33 | 35 | 1376 B | 0 | ~持平 |
| `vertical_gray<5,u8>` | 33 | 40 | 2304 B | 0 | -18% |
| `rgb_shared (STANDARD)` | 48 | — | 动态 | 0 | — |
| `adaptive_rgb` | 54 | — | 动态 | 0 | — |

**sm_110 编译器的重大改进**：

- **TEMPLATE RGB kernel 寄存器从 64 降到 28**（Opt G 后 32x8 block，之前 16x16 时为 23）：Blackwell 编译器对循环展开后的代码有更强的优化能力，大幅降低了寄存器需求
- **TEMPLATE Gray kernel 从 63 降到 21**（之前 16x16 时为 19）：同理
- Separable kernels 变化不大（循环体较短，编译器优化空间有限）
- 所有 kernel **零 spill**

> ncu 实测的 TEMPLATE kernel occupancy（97.8%）与 ptxas 理论分析（100%）吻合；SEPARABLE kernel（62-63%）与理论（66.7%）也基本一致。

### 11.4 Roofline 模型分析

Thor SM 频率 1.572 GHz, 20 SMs。基于 ncu 实测数据构建 Roofline:

#### TEMPLATE 模式 (RGB, R=5)

| 指标 | 值 | 来源 |
|------|---:|------|
| Kernel 耗时 | 4.975 ms | ncu |
| SM 吞吐量 | **88.83%** of peak | ncu SOL |
| L1/TEX 吞吐量 | 87.28% | ncu SOL |
| L2 吞吐量 | 3.50% | ncu SOL |
| DRAM 吞吐量 | ~0% | ncu (数据 fit in L2) |
| 实测 Occupancy | 97.76% | ncu |
| IPC | 3.55 inst/cycle | ncu |

**诊断**：SM 88.8% >> L2 3.5%，**纯 Compute-bound**。工作集（24.9 MB 输入 + 24.9 MB 输出）小于 L2 容量（32 MB），DRAM 流量几乎为零。

#### SEPARABLE 模式 (RGB, R=5)

| 指标 | Horizontal | Vertical |
|------|:----------:|:--------:|
| SM 吞吐量 | 64.87% | 67.28% |
| Memory 吞吐量 | 57.09% | 34.16% |
| L2 吞吐量 | 34.44% | 18.20% |
| 实测 Occupancy | 62.09% | 63.10% |
| IPC | 2.56 | 2.70 |

**诊断**：Horizontal 处于 **Compute + Memory 均衡** 状态（SM 65% / Mem 57%），Vertical 偏 **Compute-bound**（SM 67% / Mem 34%）。相比 TEMPLATE（SM 89%），两者的 SM 利用率较低，原因是 occupancy 不足（62-63% vs 97.8%）和全局内存合并效率差（22-33%）。

#### Roofline 图示

```
SM Throughput (%)
  100 ┤
   89 ┤ ··[TEMPLATE]·····················  (97.8% occ, 3.55 IPC)
      │
   67 ┤ ·········[SEP-V]                   (63.1% occ, 2.70 IPC)
   65 ┤ ········[SEP-H]                    (62.1% occ, 2.56 IPC)
      │
      │     L2 Throughput:
      │     TEMPLATE: 3.5%  (数据 fit in L2 32MB)
      │     SEP-H:   34.4%  (中间 float 缓冲读写)
      │     SEP-V:   18.2%
      └────────────────────────────────────
        所有 kernel 均为 Compute-bound (SM >> Memory)
```

### 11.5 瓶颈诊断与优化方向

```
Thor TEMPLATE 模式端到端 ~5.5 ms
           ┌──────┬──────────────────────┐
           │ 0.41 │   Kernel 4.98 ms     │
           └──────┴──────────────────────┘
            7% 拷贝│      93% 计算

Thor SEPARABLE 模式端到端 ~3.4 ms
           ┌──────┬────────────────────┐
           │ 0.42 │   H+V 3.0 ms       │
           └──────┴────────────────────┘
           12% 拷贝│     88% 计算
```

#### 基于 ncu 数据的优化优先级（含实施状态）

| 优先级 | 优化目标 | ncu 依据 | 预估收益 | 状态 |
|:------:|---------|---------|:--------:|:----:|
| ★★★ | **消除 smem bank conflict (TEMPLATE)** | 1.97x conflict → 2.3% | **10-20%** | **Opt G: 已完成** (-97.6% conflict) |
| ★★★ | **改善全局内存合并 (SEPARABLE)** | 68% 冗余 → 29-47% | **20-40%** | **Opt H: 已完成** (SoA 中间缓冲区) |
| ★★★ | **降低 Separable 寄存器** | Occupancy 62% → 97.5% | **10-20%** | **Opt K: 已完成** (regs 63→40, **-10.9%**) |
| ★★☆ | **促进 FMA 融合 (SEPARABLE)** | 37% 非融合 FP32 | **5-10%** | **Opt I: 无效** (编译器已自动融合) |
| ★☆☆ | **消除 Horizontal 分支发散** | 分支效率 83.38% | **<5%** | 待实验 |

### 11.6 Thor 上的数据波动分析

Jetson 平台 benchmark 的 stddev 明显高于独显（RTX 4060 stddev ~0.1-0.3ms vs Thor stddev ~1-7ms）。原因：

1. **DVFS（动态频率电压调节）**：Jetson 根据温度和功耗动态调整 GPU 频率，导致连续 run 之间性能波动
2. **统一内存竞争**：CPU 和 GPU 共享内存带宽，后台进程的内存活动会影响 GPU 性能
3. **热管理**：嵌入式平台散热条件有限，长时间运行后降频

因此，**min 值比 avg 值更能反映 GPU 的峰值能力**，而 avg 值反映了真实工作负载中的可预期性能。ncu 采集的单次 kernel 数据（profiling 模式下 DVFS 影响极小）最为准确。

---

## 十二、跨平台对比分析

### 12.1 平台规格对比

| 规格 | Jetson AGX Thor | RTX 4060 |
|------|:---------------:|:--------:|
| GPU 架构 | Blackwell (sm_110) | Ada Lovelace (sm_89) |
| SM 数量 | 20 | 24 |
| 显存/统一内存 | 128 GB LPDDR5x (共享) | 8 GB GDDR6 (独立) |
| L2 Cache | 32 MB | — |
| Shared Memory/SM | 228 KB | 100 KB |
| 寄存器/SM | 65536 | 65536 |
| CPU-GPU 互联 | 统一内存 (无 PCIe) | PCIe 4.0 x8 (WSL2) |
| 平台类型 | 嵌入式 SoC | 桌面独显 |
| CUDA 版本 | 13.0 | 13.1 |

### 12.2 4K RGB 性能对比

| 模式 | Thor Avg (ms) | Thor Min (ms) | RTX 4060 Avg (ms) | RTX 4060 Min (ms) |
|------|:------------:|:------------:|:------------------:|:-----------------:|
| TEMPLATE | 23.20 | 16.86 | 6.67 | 6.33 |
| SEPARABLE | 18.88 | 12.84 | 5.61 | 5.41 |
| STANDARD | 28.57 | 9.22 | 8.34 | 7.96 |
| ADAPTIVE | 21.07 | 12.89 | 6.95 | 6.80 |

> 注：RTX 4060 数据含 OpenCV CUDA 基线（11.78ms），Thor 上因 OpenCV 无 CUDA 模块而未包含。

### 12.3 nsys/ncu Kernel 耗时对比

| Kernel | Thor (ms) | RTX 4060 (ms) | Thor/RTX 4060 |
|--------|:---------:|:------------:|:-------------:|
| `rgb_template<5,u8,u8>` | 4.98 | 3.36 | 1.48x |
| `horizontal_rgb<5,u8>` | 1.52 | 0.79 | 1.92x |
| `vertical_rgb<5,u8>` | 1.48 | 0.72 | 2.06x |
| `cudaMemcpy` (H2D+D2H) | 0.41 | 3.94 | **0.10x** |

#### ncu 指标对比 (TEMPLATE RGB kernel)

| 指标 | Thor (sm_110) | RTX 4060 (sm_89) |
|------|:------------:|:----------------:|
| SM Throughput | **88.83%** | ~20% (推算) |
| 实测 Occupancy | **97.76%** | ~67% |
| Registers/Thread | 23 | 64 |
| IPC (active) | 3.55 | — |
| L1 命中率 | 68.53% | — |
| L2 命中率 | 73.31% | — |
| Smem bank conflict | 1.97x | — |
| 首要 stall | short_scoreboard | — |

**分析**：

- **Kernel 计算**：Thor 比 RTX 4060 慢 1.5-2x，主要因为 SM 数量更少（20 vs 24）且核心频率较低
- **数据传输**：Thor 仅需 0.41ms（统一内存，~120 GB/s），RTX 4060 需 3.94ms（PCIe DMA，~12 GB/s），Thor 快 **10x**
- **SM 利用率**：Thor 88.8% 远高于 RTX 4060 的 ~20%，说明 Blackwell 编译器的寄存器优化（64->23）和高 occupancy（97.8% vs 67%）极大提升了计算效率
- **尽管频率和 SM 数更少，Thor 的 SM 效率更高**，每个 SM 的工作负载更饱和

### 12.4 编译器差异的影响

sm_110 (Blackwell) 编译器对 TEMPLATE kernel 的寄存器优化是最大惊喜：

| Kernel | sm_110 Regs | sm_89 Regs | 差异 | Occupancy (sm_110) | Occupancy (sm_89) |
|--------|:-----------:|:----------:|:----:|:------------------:|:-----------------:|
| rgb_template | **23** | 64 | -64% | **100%** | 67% |
| gray_template | **19** | 63 | -70% | **100%** | ~67% |

尽管 occupancy 大幅提升，但 kernel 耗时反而更长（4.98ms vs 3.36ms）。这说明 RTX 4060 上该 kernel 的瓶颈不在 occupancy，而在于核心频率和 SM 数量。寄存器降低带来的 occupancy 提升无法弥补硬件规格差异。

### 12.5 统一内存架构对优化策略的影响

| 优化手段 | 独显 (RTX 4060) | 统一内存 (Thor) | 说明 |
|---------|:---------------:|:---------------:|------|
| cudaHostRegister | +7% | 无意义 | 统一内存无需 DMA，page-lock 不改变传输方式 |
| Strip Pipeline | 理论 2-3x | 无意义 | 无传输可重叠 |
| FP16 中间缓冲 | +2.7% | 有效 | 统一内存带宽有限，减少数据量仍有收益 |
| Shared Memory | 有效 | 有效 | 减少全局内存访问对两种架构都关键 |
| LUT 优化 | 有效 | 有效 | 消除 expf 对两种架构都关键 |
| Template Unroll | 有效 | 有效 | sm_110 上编译器优化更激进 |

**结论**：统一内存架构下，所有传输相关优化（page-lock、strip pipeline、multi-stream overlap）都失去意义。优化焦点应完全放在 **kernel 计算效率** 上。

---

## 十三、参考资料分析与待实验优化思路

> 来源：[xytroot/Bilateral-Filter](https://github.com/xytroot/Bilateral-Filter) 与 [OpenCV CUDA bilateral_filter.cu](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaimgproc/src/cuda/bilateral_filter.cu)

### 13.1 参考实现对比

#### xytroot/Bilateral-Filter（教学级实现）

| 技术 | xytroot | 本项目 | 对比 |
|------|---------|--------|------|
| Texture Memory | `tex2D()` + `cudaBindTexture2D` | 无（Shared Memory） | 见 13.2-A |
| 空间权重 | 1D `__constant__` 数组，kernel 内合成 2D | 预计算完整 2D LUT | 本项目更优（1 次 vs 2 次查表） |
| 颜色权重 | kernel 内实时 `__expf()` | 256 元素 color LUT | **本项目远优** |
| Shared Memory | 无 | Tile + halo 协作加载 | 本项目更优 |
| 内存管理 | 每次 `cudaMalloc` / `cudaFree` | 持久化 GPU buffer | 本项目更优 |

#### OpenCV CUDA bilateral_filter.cu

| 技术 | OpenCV CUDA | 本项目 | 意义 |
|------|-------------|--------|------|
| 圆形窗口裁剪 | `if (space2 > r2) continue` | LUT 预置零 + early continue | 已实现（效果更优） |
| 颜色距离 | `norm_l1` + `exp()` 实时计算 | 预计算 LUT | **本项目更快** |
| Shared Memory | **无** | 有 | **本项目更优** |

### 13.2 Thor 平台上的进一步优化方向（基于 ncu 数据）

基于 11.2 节 ncu 硬件计数器分析和 11.5 节瓶颈诊断，Thor 上的优化方向与独显平台截然不同：

#### 已完成的 ncu 驱动优化

| 优化 | 实施方案 | ncu 前后对比 | 实测效果 |
|------|---------|:----------:|---------|
| **Opt G: smem bank conflict** | Block 16x16 → 32x8 | conflict: 50% → **2.3%** | SM +0.87pp, latency 持平 |
| **Opt H: global coalescing** | 中间缓冲区 AoS → SoA | uncoalesced: 68% → **29%** (V), 69% → **47%** (H) | latency 持平 |
| **Opt I: FMA fusion** | 显式 `fmaf()` | fused/non-fused 比率不变 | **无效**（编译器已自动融合） |
| **Opt K: register pressure** | `launch_bounds(256, 6)` | regs: 63 → **40**, occ: 62% → **97.5%** | **4K RGB -10.9%** |

> **Opt G 深度分析**：最初尝试 smem row padding（TILE_W 从 26 改为 27，奇数 stride），但 ncu 验证后发现 conflict 仍为 50%。深入分析发现根因是 **warp 跨行**（16x16 block 中 warp 包含两行线程），而非行内 stride 问题。最终改为 32x8 block（warp 全在一行），彻底消除了 load conflict。

> **Opt I 教训**：nvcc `-O3` 的 FMA 融合已经接近最优。ncu 报告的 "37% 非融合 FP32" 指的是独立的 FADD/FMUL（如 `weight_sum += w`），它们没有配对指令可融合，不是编译器遗漏。

#### TEMPLATE 模式剩余优化空间

| 优先级 | 优化手段 | ncu 依据 | 预期收益 |
|:------:|---------|---------|:--------:|
| ★★☆ | **Spatial LUT 搬到 smem** | constant cache 99.99% hit，但 warp 内不同地址仍序列化 | **5-10%** |
| ★☆☆ | **cudaMallocManaged 零拷贝** | cudaMemcpy 仅占 7%，但 0.41ms 仍可省 | **~5%** |

#### SEPARABLE 模式剩余优化空间

| 优先级 | 优化手段 | ncu 依据 | 预期收益 | 实测结果 |
|:------:|---------|---------|:--------:|:--------:|
| ~~★★★~~ | ~~降低寄存器用量~~ | ~~Occupancy 62%→97.5%~~ | ~~**-10.9%**~~ | **Opt K: 已完成** |
| ~~★★☆~~ | ~~u8 RGB 输入向量化~~ | ~~输入仍有 40% uncoalesced~~ | ~~10-20%~~ | **Opt L: 取消**（sectors/req=2.98≈理论极限） |
| ~~★★☆~~ | ~~Fused H+V Kernel~~ | ~~消除中间 buffer~~ | ~~10-15%~~ | **Opt M: -31~34%**（计算膨胀抵消带宽节省） |
| ★★☆ | **FP16 中间缓冲区** | H↔V buffer 带宽减半 | 5-10% | **Opt N: Gray -7~10%, RGB -1.7%** |
| ★☆☆ | **消除 Horizontal 分支发散** | 83.38% 分支效率, 3267 divergent branches | **<5%** | 未实施 |

#### 跨模式通用优化

| 优先级 | 优化手段 | 说明 | 预期收益 | 实测结果 |
|:------:|---------|------|:--------:|:--------:|
| ~~★★☆~~ | ~~FP16 全量计算~~ | ~~Thor FP16 吞吐 2x~~ | ~~10-20%~~ | **Opt N2: -8.4%**（标量 half 无 2x，转换开销高） |
| ★☆☆ | **L2 Cache 持久化** | Thor 有 32MB L2, 当前 TEMPLATE 数据已 fit in L2 (hit 73%) | **<5%** | 未实施 |
| ★☆☆ | **Spatial LUT 搬到 smem** | constant cache warp 序列化 | 5-10% | 未实施 |

> **结论**：SEPARABLE 模式的主要优化空间已基本挖掘完毕。SM throughput 达 75-81%，occupancy 97%+，bank conflict 0%，中间缓冲区已 SoA + FP16。剩余优化（分支发散、L2 持久化、spatial LUT）预期收益均 < 5%，投入产出比低。当前 SEPARABLE_FP16 (MODE=3) 以 2.97ms/4K RGB（2741 MP/s）达到接近硬件极限的性能。

> **注意**：独显平台上优先级最高的传输优化（strip pipeline、multi-stream、cudaHostRegister）在 Thor 上完全不适用——传输仅占端到端 5-12%。

### 13.3 总结与展望

本项目在两个截然不同的 GPU 平台上验证了 CUDA 双边滤波的优化效果，经历了 15 个优化实验（其中 7 个成功保留，4 个实验失败但提供了宝贵的负面结论，4 个回退）。

**已达成目标**：

| 目标 | 要求 | Thor 实测 (Opt G/H/K/N) | RTX 4060 实测 |
|------|------|:---------:|:------------:|
| MAE | < 1.0 | 0.12~0.61 | 0.15~0.61 |
| PSNR | > 40 dB | 48.28~57.00 | 48.28~56.18 |
| vs OpenCV CPU | 显著加速 | **9~39x** | **7~20x** |
| vs OpenCV CUDA | 超越 | N/A（Thor 无 OCV CUDA） | **1.68~4.43x** |

**峰值性能**：SEPARABLE_FP16 (MODE=3) 在 Thor 上达到 **4K RGB 2.97ms (2741 MP/s)**，**4K Gray 1.30ms (5915 MP/s)**。SM throughput 75-81%，occupancy 97%，接近硬件极限。

**最有价值的优化手段（跨平台通用）**：

1. **Color Weight LUT (3x)**：消除 `expf` 调用，对所有平台收益最大
2. **Shared Memory (3-5x)**：减少全局内存访问，对所有平台关键
3. **Template Unroll + Circular Window DCE (+13%~+65%)**：编译器在编译期消除 33% 圆外迭代
4. **Separable Approximation (O(r) vs O(r^2))**：算法级优化，平台无关
5. **Block 32x8 (smem bank conflict -97.6%)**：ncu 驱动的微架构优化
6. **SoA 中间缓冲区 (global uncoalesced -39pp)**：ncu 驱动的内存布局优化
7. **Launch Bounds (256,6) (occupancy +35pp, 4K -10.9%)**：ncu 驱动的寄存器压力优化

**关键负面结论（同样重要）**：

| 失败实验 | 预期收益 | 实测 | 根因分析 |
|---------|:--------:|:----:|---------|
| Fused H+V Kernel | 10-15% | **-31~34%** | 统一内存下中间 buffer 成本仅 ~10%，不值得增加 block 计算复杂度 |
| FP16 全量计算 | 10-20% | **-8.4%** | 标量 `__half` 无 2x 优势（仅 `__half2` 有），float↔half 转换开销高 |
| u8 RGB 向量化 | 10-20% | 取消 | sectors/req=2.98 已是 RGB 3B/pixel AoS 格式的理论极限（75%） |

**平台特异性认知**：

| 认知 | 独显 (PCIe) | 统一内存 (Jetson) |
|------|:-----------:|:-----------------:|
| 主瓶颈 | H2D/D2H 传输 (46-68%) | Kernel 计算 (90-95%) |
| 传输优化价值 | 极高 | 无意义 |
| Kernel 优化边际收益 | 中等（被传输稀释） | **极高**（直接反映到端到端） |
| 编译器优化 | sm_89 保守 | sm_110 激进（寄存器 -64%） |
| Fused kernel 价值 | 可能有效（传输成本高） | 无效（中间 buffer 成本低） |
| FP16 策略 | 仅中间缓冲区有效 | 同左（标量 half 无优势） |
