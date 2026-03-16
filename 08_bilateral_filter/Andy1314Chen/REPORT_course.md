# CUDA 双边滤波优化实验报告

**题目**：CUDA 双边滤波优化实验与性能分析  
**课程**：并行计算 / GPU 编程  
**作者**：Andy1314Chen  
**日期**：2026-02-27  
**项目路径**：`08_bilateral_filter/Andy1314Chen`

---

## 目录

1. [引言](#1-引言)
2. [算法原理与任务定义](#2-算法原理与任务定义)
3. [CUDA 实现模式设计](#3-cuda-实现模式设计)
4. [实验环境与评测方法](#4-实验环境与评测方法)
5. [基线实现与瓶颈分析](#5-基线实现与瓶颈分析)
6. [优化策略与逐步实验](#6-优化策略与逐步实验)
7. [性能结果与质量验证](#7-性能结果与质量验证)
8. [Profiler 深度分析](#8-profiler-深度分析)
9. [失败实验与优化边界](#9-失败实验与优化边界)
10. [跨平台对比分析](#10-跨平台对比分析)
11. [结论与展望](#11-结论与展望)
12. [附录：完整数据图表](#12-附录完整数据图表)

---

## 1. 引言

双边滤波是一种经典的非线性保边平滑滤波器，在 VR 头显弱光降噪、医学图像处理等领域有广泛应用。然而，其计算复杂度为 $O((2r+1)^2)$，当滤波半径 $r=5$ 时每个像素需进行 121 次邻域访问和权重计算，对 4K（3840×2160）@60fps 实时处理构成极大挑战。

本文目标：在 CUDA 平台上实现高性能双边滤波，满足以下指标：
- **吞吐量**：4K@60fps 即 ≥498 MP/s
- **正确性**：MAE < 1.0、PSNR > 40 dB（以 OpenCV CPU `bilateralFilter` 为参考）
- **性能**：相对 OpenCV CPU 显著加速，并超越 OpenCV CUDA

实验经历了 12 个迭代版本和 15 个优化实验（含 5 个失败实验），最终在两个 GPU 平台上验证了优化效果。

---

## 2. 算法原理与任务定义

### 2.1 双边滤波公式

$$
BF[I]_p = \frac{1}{W_p} \sum_{q \in \mathcal{S}} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q
$$

其中：
- $G_{\sigma_s}(\|p-q\|) = \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right)$ 为**空间高斯权重**
- $G_{\sigma_r}(|I_p-I_q|) = \exp\left(-\frac{|I_p-I_q|^2}{2\sigma_r^2}\right)$ 为**值域高斯权重**
- $W_p = \sum_{q} G_{\sigma_s} \cdot G_{\sigma_r}$ 为归一化因子

在平坦区域，颜色相似使权重近似空间高斯，实现充分平滑；在边缘处，颜色差异大使值域权重趋零，避免跨边缘模糊。

### 2.2 计算量分析

对于 4K RGB 图像（3840×2160×3），radius=5：
- 每个像素：$(2 \times 5+1)^2 = 121$ 次邻域访问
- 每次访问：1 次空间权重 + 1 次值域权重（含 `expf`）+ 加权累加
- 全图：$8.3M \times 121 \approx 10$ 亿次浮点运算

### 2.3 代码结构

```
src/
  bilateral_filter_cuda.cu    # 全部 CUDA kernel（6 种模式）、LUT、pipeline
  bilateral_filter_cpu.cpp    # CPU 参考实现
  bilateral_filter_opencv.cpp # OpenCV CPU + CUDA 封装
  main.cpp                    # CLI 入口（--bench/--cuda/--compare-all）
  image_io.cpp                # 二进制 raw 图像 I/O
include/
  bilateral_filter_cuda.cuh   # CUDA 滤波声明
  bilateral_filter.h          # CPU 滤波声明
  image_io.h                  # ImageData/FilterParams 结构体
```

---

## 3. CUDA 实现模式设计

本项目实现了 6 种 CUDA 滤波模式，通过环境变量 `BILATERAL_MODE` 在运行时切换。各模式在算法复杂度、编译器优化能力、精度和适用场景上有本质差异。以下从代码层面逐一分析。

### 3.1 STANDARD 模式（MODE=0）—— 运行时半径

STANDARD 是最通用的 fallback 实现：半径作为**运行时参数**传入 kernel，shared memory 大小通过 `extern __shared__` 动态分配。

```cpp
// bilateral_filter_cuda.cu:1031-1034 — 运行时半径，动态 shared memory
__global__ void k_bilateral_filter_shared(
    const InT* __restrict__ input, OutT* __restrict__ output,
    int width, int height, int radius) {
    extern __shared__ float smem[];  // 大小在 launch 时指定
    ...
    #pragma unroll 4  // 仅提示展开因子 4（循环界不确定）
    for (int dy = -radius; dy <= radius; ++dy) { ... }
}
```

**核心特征**：
- 支持**任意半径值**，无需预编译
- 循环界为运行时变量，编译器**无法完全展开**，仅按 `#pragma unroll 4` 部分展开
- RGB 版本对三通道使用**独立权重**（每通道单独查 LUT、独立累加），精度最高但 LUT 访问量为 TEMPLATE 的 3 倍

```cpp
// bilateral_filter_cuda.cu:1160-1180 — STANDARD RGB: 三通道独立权重
float cw_r = d_color_lut[diff_r];  // R 通道权重
float cw_g = d_color_lut[diff_g];  // G 通道权重
float cw_b = d_color_lut[diff_b];  // B 通道权重
sum_r += nr * (spatial_weight * cw_r);  // 各通道独立加权
sum_g += ng * (spatial_weight * cw_g);
sum_b += nb * (spatial_weight * cw_b);
```

**适用场景**：半径不在预编译列表（3/5/7/9/10）中时的通用回退。

### 3.2 TEMPLATE 模式（MODE=1）—— 编译期半径

TEMPLATE 是性能最均衡的模式：将半径作为 **C++ 模板参数**，使编译器获得完整循环界信息。

```cpp
// bilateral_filter_cuda.cu:74-76 — 编译期半径
template <int RADIUS, typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_gray_template(...) {
    constexpr int TILE_W = BLOCK_X + 2 * RADIUS;  // 编译期常量
    ...
    #pragma unroll  // 编译器可完全展开（RADIUS 已知）
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        #pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) { ... }
    }
}
```

**与 STANDARD 的关键差异**：

| 维度 | STANDARD | TEMPLATE |
|------|----------|----------|
| 半径 | 运行时参数 `int radius` | 编译期常量 `template<int RADIUS>` |
| Shared memory | `extern __shared__` 动态 | `__shared__ float smem[编译期大小]` |
| 循环展开 | `#pragma unroll 4`（部分） | `#pragma unroll`（**完全展开**） |
| 编译器优化 | 有限（循环界未知） | **DCE + 寄存器优化 + 指令调度** |
| RGB 权重 | 三通道独立（3 次 LUT） | **均值近似**（1 次 LUT） |

TEMPLATE 的 RGB kernel 使用**单一色彩权重近似**——三通道差值的均值作为查表索引，LUT 查找从 3 次降为 1 次：

```cpp
// bilateral_filter_cuda.cu:218-224 — TEMPLATE RGB: 单色权重近似
int diff = static_cast<int>(
    (fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b))
    * (1.0f / 3.0f) + 0.5f);
float w = spatial_weight * d_color_lut[diff]; // 1 次查表，3 通道共享
```

**圆形窗口 DCE**：由于 RADIUS 是编译期常量，`#pragma unroll` 完全展开 121 次迭代后，编译器在编译期确定哪 40 个位置的 `d_spatial_lut` 恒为 0，通过 Dead Code Elimination 直接删除这些迭代，等效于仅执行 81 次。

运行时通过 `switch(radius)` 分发到预编译的模板实例：

```cpp
// 支持 radius = 3, 5, 7, 9, 10 五种预编译版本
switch (radius) {
    case 3: launch_u8_gray<3>(d_in, d_out, w, h, s); break;
    case 5: launch_u8_gray<5>(d_in, d_out, w, h, s); break;
    case 7: launch_u8_gray<7>(d_in, d_out, w, h, s); break;
    ...
}
```

### 3.3 SEPARABLE 模式（MODE=2）—— 分离近似

SEPARABLE 是**算法级优化**：将 2D 双边滤波近似为水平 + 垂直两次 1D 滤波。

严格来说，双边滤波因值域权重依赖像素值而**不可分离**。但实践中近似分离的质量损失很小（MAE 0.45 vs STANDARD 0.48），且复杂度从 $O((2r+1)^2)$ 降至 $O(2 \times (2r+1))$——r=5 时从 121 次降到 22 次。

**水平 pass**：每行加载到 shared memory，沿行方向滤波：

```cpp
// bilateral_filter_cuda.cu:246-296 — 水平 kernel
template <int RADIUS, typename InT, typename TmpT>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM_SEP)
k_bilateral_horizontal_gray(...) {
    __shared__ float smem[BLOCK_Y][TILE_W_PAD]; // 每行独立
    // 加载行 + halo
    for (int i = tx; i < TILE_W; i += BLOCK_X) {
        smem[ty][i] = static_cast<float>(input[gy * width + gx]);
    }
    __syncthreads();
    // 1D 滤波（仅沿 dx 方向）
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
        float neighbor = smem[ty][lx + dx];
        float spatial_weight = d_spatial_lut[RADIUS * LUT_WIDTH + (dx + RADIUS)];
        ...
    }
}
```

**垂直 pass**：每列加载到 shared memory，沿列方向滤波：

```cpp
// bilateral_filter_cuda.cu:298-347 — 垂直 kernel
template <int RADIUS, typename TmpT, typename OutT>
__global__ void k_bilateral_vertical_gray(...) {
    __shared__ float smem[TILE_H][BLOCK_X]; // 每列独立
    // 1D 滤波（仅沿 dy 方向）
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        float neighbor = smem[ly + dy][tx];
        float spatial_weight = d_spatial_lut[(dy + RADIUS) * LUT_WIDTH + RADIUS];
        ...
    }
}
```

**与 TEMPLATE 2D 的关键差异**：

| 维度 | TEMPLATE (2D) | SEPARABLE (H+V) |
|------|---------------|-----------------|
| 复杂度 | $O((2r+1)^2) = 121$ 次 | $O(2 \times (2r+1)) = 22$ 次 |
| Kernel 数 | 1 个 | 2 个（H + V） |
| 中间缓冲区 | 无 | float 或 FP16（SoA 布局） |
| Shared memory | 2D tile (H×W) | 1D 行或列 |
| 精度 | 精确 2D 滤波 | 近似（可分离假设） |
| launch_bounds | `(256, 4)` | `(256, 6)`（更激进） |

### 3.4 SEPARABLE_FP16 模式（MODE=3）—— FP16 中间缓冲

在 SEPARABLE 基础上，将两趟之间的中间缓冲区从 `float`（4B）改为 `__half`（2B），**全局内存带宽减半**。计算仍在 FP32 进行，精度无损。

```cpp
// bilateral_filter_cuda.cu:1332-1335
// H kernel: uint8 → smem(float) → compute(float) → output(__half)
// V kernel: input(__half) → smem(float) → compute(float) → output(uint8)
// 仅 H↔V 传输使用 FP16，计算精度不受影响。
```

ncu 验证：V kernel LD sectors/request 从 4.00 降至 2.00（**-50%**），Gray 模式端到端提速 **7-10%**。

### 3.5 ADAPTIVE 模式（MODE=4）—— 自适应半径

ADAPTIVE 是**质量优先**的模式：先用 Sobel 算子计算逐像素梯度，再将梯度映射为半径——平坦区域用大半径（充分平滑），边缘区域用小半径（保留细节）。

**第一步：梯度计算 → 半径映射**

```cpp
// bilateral_filter_cuda.cu:1491-1506 — Sobel 梯度 → radius
float gx = -p00 + p02 - 2.0f*p10 + 2.0f*p12 - p20 + p22;
float gy = -p00 - 2.0f*p01 - p02 + p20 + 2.0f*p21 + p22;
float grad = sqrtf(gx*gx + gy*gy);
// 线性插值：grad=0 → r_max，grad≥threshold → r_min
float t = fminf(grad * inv_grad_threshold, 1.0f);
int r = __float2int_rn(r_max - t * (r_max - r_min));
radius_map[y * width + x] = static_cast<uint8_t>(r);
```

**第二步：使用 per-pixel radius 的双边滤波**

```cpp
// bilateral_filter_cuda.cu:1551-1588 — 自适应 kernel
k_bilateral_adaptive_gray(..., const uint8_t* radius_map, ...) {
    // shared memory 按 r_max 分配（保证所有线程可访问最大邻域）
    const int tile_w = BLOCK_X + 2 * r_max;
    ...
    const int my_radius = static_cast<int>(radius_map[y * width + x]);
    for (int dy = -my_radius; dy <= my_radius; ++dy) { ... }
}
```

**与其他模式的关键差异**：

| 维度 | TEMPLATE/STANDARD | ADAPTIVE |
|------|-------------------|----------|
| 半径 | 全图统一 | **逐像素不同** |
| 预处理 | 无 | 额外 Sobel 梯度 kernel |
| Warp divergence | 无（循环界一致） | **有**（同 warp 内不同线程循环次数不同） |
| Shared memory | 按实际 radius 分配 | 按 **r_max** 分配（最坏情况） |
| MAE | 0.48-0.60 | **0.40**（最优） |
| 性能 | 较快 | 较慢（梯度 pass + divergence 开销） |

### 3.6 FUSED 模式（MODE=5）—— 融合 H+V

实验性模式：将 SEPARABLE 的两个 kernel 融合为单 kernel，消除中间缓冲区的全局内存读写。

```
单 kernel 三阶段：
Phase 0: 加载 2D halo → smem_raw
Phase 1: 对全部行做水平滤波 → smem_h（smem 内，无全局写入）
Phase 2: 从 smem_h 做垂直滤波 → 输出
```

实测性能倒退 31-34%（见第 9 章失败实验），已标记为实验性。

### 3.7 模式对比总结

下表为 **Jetson AGX Thor** 实测数据（统一内存，无 PCIe 瓶颈，kernel 性能直接反映端到端）。OpenCV CPU 82 ms = 1.00x。

| 模式 | 环境变量 | 复杂度 | MAE | Avg (ms) | Min (ms) | vs OCV CPU | vs OCV CUDA | 核心特点 |
|------|---------|:---:|---:|---:|---:|---:|---:|---------|
| **SEP_FP16** | **`MODE=3`** | **$O(r)$** | **0.46** | **3.01** | **2.95** | **27.2x** | **4.00x** | **SEPARABLE + FP16 中间缓冲，最快** |
| SEPARABLE | `MODE=2` | $O(r)$ | 0.45 | 3.06 | 2.99 | 26.8x | 3.94x | 水平+垂直分离近似 |
| TEMPLATE | `MODE=1` | $O(r^2)$ | 0.60 | 5.48 | 5.44 | 15.0x | 2.23x | 编译期半径，完全展开 + DCE |
| ADAPTIVE | `MODE=4` | $O(r_{avg}^2)$ | **0.40** | 6.17 | 6.11 | 13.3x | 1.94x | Sobel 自适应半径，最高精度 |
| STANDARD | `MODE=0` | $O(r^2)$ | 0.48 | 9.41 | 9.26 | 8.70x | 1.29x | 运行时半径，三通道独立权重 |
| OpenCV CUDA | — | — | 0.00 | 12.07 | 11.79 | 6.79x | 1.00x | GPU 参考基线 |
| **OpenCV CPU** | — | — | — | **82** | — | **1.00x** | — | **CPU 基线** |

---

## 4. 实验环境与评测方法

### 4.1 硬件平台

| 规格 | RTX 4060 (桌面) | Jetson AGX Thor (嵌入式) |
|------|:---:|:---:|
| GPU 架构 | Ada Lovelace (sm_89) | Blackwell (sm_110) |
| SM 数量 | 24 | 20 |
| 显存/统一内存 | 8 GB GDDR6 (独立) | 128 GB LPDDR5x (共享) |
| L2 Cache | 24 MB | 32 MB |
| Shared Memory/SM | 100 KB | 228 KB |
| CPU-GPU 互联 | PCIe 4.0 x8 (WSL2) | 统一内存 (无 PCIe) |
| CUDA | 13.1 | 13.0 |
| OpenCV | 4.13.0 (with CUDA) | 4.x (with CUDA) |

### 4.2 评测方法

- **Benchmark 流程**：5 次 warmup + 50 次计时，报告 mean ± stddev
- **测试数据**：4K/1080p × RGB/Gray，参数 radius=5, σ_s=3.0, σ_c=30.0
- **基线口径**：
  - **主基线**：OpenCV CPU `bilateralFilter`（正确性参照 + 性能基准）
  - **次基线**：OpenCV CUDA `cv::cuda::bilateralFilter`（GPU 性能参照）
- **工具**：`ncu --set full`（硬件计数器）、`nsys profile`（时间线）

---

## 5. 基线实现与瓶颈分析

### 5.1 OpenCV CPU 基线

本项目以 OpenCV `cv::bilateralFilter`（CPU 版本）作为 **主基线**——它是工业级优化的参考实现，也是正确性验证的金标准。

4K RGB 耗时 **82 ms**（~101 MP/s），作为 1.00x 基线。

项目同时提供了一份朴素 CPU 实现（`bilateral_filter_cpu.cpp`），用于展示算法逻辑。其核心循环与 CUDA naive kernel 对应：

```cpp
// bilateral_filter_cpu.cpp — 朴素三重循环（仅作算法参照，不作为性能基线）
for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
        float spatial_weight = expf((dx*dx + dy*dy) * spatial_coeff);
        float color_dist = neighbor_val - center_val;
        float color_weight = expf(color_dist * color_dist * color_coeff);
        float weight = spatial_weight * color_weight;
        sum += neighbor_val * weight;
        weight_sum += weight;
    }
}
output[idx] = sum / weight_sum;
```

> 该朴素 CPU 实现在 1080p RGB 上耗时约 6918 ms，比 OpenCV CPU（~27 ms）慢约 **256 倍**（OpenCV 内部使用 SIMD + 缓存优化），不作为性能对比基准。

### 5.2 Naive CUDA Kernel（v1）

最初的 CUDA 实现将 CPU 逻辑直接迁移到 GPU：每个线程处理一个像素，从全局内存独立读取邻域数据，实时计算 `expf`。4K RGB 耗时 **250 ms**（33 MP/s），甚至慢于 CPU——原因在于：

### 5.3 瓶颈分析

| 瓶颈 | 分析 | 量化 |
|------|------|------|
| **全局内存冗余** | 16×16 block + r=5 时，256 线程发起 30,976 次访问，实际不重复仅 676 个像素，冗余率 **45x** | 带宽浪费 ~97% |
| **expf 调用代价** | 每像素 121 次 `expf`（~20 cycles/次），RGB 三通道 363 次 | 占 kernel 时间 ~60% |
| **warp divergence** | 边界像素的 `if (ny < 0 || ny >= height) continue` 导致同 warp 线程执行路径不一致 | 边界 block 效率下降 ~30% |
| **PCIe 传输** | 4K RGB 24.9 MB，PCIe 4.0 x8 (WSL2) 有效带宽 ~8 GB/s | H2D+D2H 约 6 ms |

---

## 6. 优化策略与逐步实验

### 6.1 v2: Shared Memory 协作加载（+42%）

**动机**：消除邻域的全局内存重复访问。

**实现**：block 内线程协作将 tile（含 halo 区域）加载到 shared memory，边界用 clamp 策略避免分支：

```cpp
// bilateral_filter_cuda.cu:96-107
#pragma unroll
for (int i = thread_id; i < TILE_SIZE_LOG; i += threads_per_block) {
    int gx = blockIdx.x * BLOCK_X + sx - RADIUS;
    gx = max(0, min(width - 1, gx));  // clamp 边界
    gy = max(0, min(height - 1, gy));
    smem[sy * TILE_W_PAD + sx] = static_cast<float>(input[gy * width + gx]);
}
__syncthreads();  // 确保数据就绪
```

shared memory 大小为 $(blockDim + 2r)^2$，访问延迟从 ~200-400 ns 降至 ~5 ns。

| 版本 | 4K RGB (ms) | 吞吐量 (MP/s) | vs OpenCV CPU |
|------|---:|---:|---:|
| v1 naive | 250 | 33 | 0.23x |
| **v2 shared** | **176** | **47** | **0.32x** |

### 6.2 v3-v4: LUT 替代 expf（+26%, +154%）

**动机**：消除内循环中代价最高的超越函数调用。

**空间权重 LUT**（v3）：空间权重仅依赖偏移 $(dx,dy)$，预计算后存入 constant memory（warp 广播，单次事务）：

```cpp
// bilateral_filter_cuda.cu:52-53
__constant__ float d_spatial_lut[LUT_SIZE];     // (2r+1)² 个 float
__constant__ float d_color_lut[COLOR_LUT_SIZE]; // 256 个 float

// bilateral_filter_cuda.cu:1194-1214 — LUT 初始化
static void init_spatial_lut(int radius, float sigma_spatial) {
    float coeff = -0.5f / (sigma_spatial * sigma_spatial);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            lut[(dy+radius)*w + (dx+radius)] = expf((dx*dx+dy*dy) * coeff);
    cudaMemcpyToSymbol(d_spatial_lut, lut.data(), ...);
}
```

**值域权重 LUT**（v4-v5）：8-bit 图像差值范围 [0,255]，预计算 256 项查表**完全消除 expf**：

```cpp
// kernel 内：查表替代 expf
int diff = static_cast<int>(fabsf(neighbor - center) + 0.5f);
diff = min(diff, 255);
float color_weight = d_color_lut[diff];  // ~4 cycles vs expf ~20 cycles
```

ncu 实测 constant cache 命中率 **99.99%**。

| 版本 | 4K RGB (ms) | 吞吐量 (MP/s) | vs OpenCV CPU |
|------|---:|---:|---:|
| v3 spatial LUT | 140 | 59 | 0.40x |
| v4 fast math | 55 | 150 | 1.03x |
| **v5 color LUT** | **18** | **460** | **3.14x** |

> Color LUT 是单项收益最大的优化（**~3x**），每像素消除了 3 通道 × 81 邻域 = 243 次 `expf` 调用。

### 6.3 v6: Template 编译期半径（+7%）

**动机**：使编译器完全展开双重循环，消除循环控制开销，提升 ILP。

```cpp
// bilateral_filter_cuda.cu:74-75 — 模板参数化
template <int RADIUS, typename InT = float, typename OutT = float>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
k_bilateral_filter_gray_template(...) {
    #pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        #pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) { ... }
    }
}
```

运行时通过 `switch(radius)` 分发到对应模板实例化版本（支持 r=3/5/7/9/10）。

### 6.4 v7-v10: 工程级优化（持久缓冲 +71%，u8 I/O +11%，页锁定 +7%）

| 优化 | 机制 | 收益 |
|------|------|---:|
| **持久 GPU 缓冲** | 静态 `g_bufs` 结构体，仅尺寸变化时 `cudaMalloc`，LUT 参数不变则跳过上传 | **+71%** |
| **uint8 直接 I/O** | kernel 模板参数 `<InT=uint8_t, OutT=uint8_t>`，省去 host 端 u8→float→u8 转换 | **+11%** |
| **cudaHostRegister** | 将调用方堆内存注册为 page-locked，使 H2D/D2H 走 DMA 通道 | **+7%** |
| **Block 16×16** | 较好的 L1/smem 缓存利用率 | **+1%** |

### 6.5 v11: RGB 单一色彩权重（+16%）

**动机**：原始实现每邻域需 3 次 LUT 查找（R/G/B 各一次）。改用三通道均值距离，仅 1 次查找：

```cpp
// bilateral_filter_cuda.cu:218-222
int diff = static_cast<int>(
    (fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b))
    * (1.0f / 3.0f) + 0.5f);
float w = spatial_weight * d_color_lut[diff]; // 1 次 vs 3 次
```

MAE 从 0.65 略升至 0.80，仍满足 < 1.0 要求。

### 6.6 v12: 圆形窗口 Early-Continue（+13% RGB, +65% Gray）

**动机**：r=5 的方形窗口 121 个位置中，40 个角落点超出半径圆，空间权重为零。

**Phase 1**：在 `init_spatial_lut` 中预置零圆外项（零性能成本，MAE 改善 0.15~0.20）：

```cpp
// bilateral_filter_cuda.cu:1206-1207
if (dx * dx + dy * dy > r2)
    lut[...] = 0.0f;  // 33% 的位置置零
```

**Phase 2**：kernel 内 `if (spatial_weight == 0.0f) continue;`（`bilateral_filter_cuda.cu:124`）。TEMPLATE 模式下 RADIUS 为编译期常量，`#pragma unroll` 完全展开后编译器通过 **Dead Code Elimination** 直接删除这 40 次迭代的全部指令。

| 模式 | Before (ms) | After (ms) | 提升 |
|------|---:|---:|---:|
| TEMPLATE RGB | 7.36 | **6.53** | **+13%** |
| TEMPLATE Gray | 4.97 | **3.01** | **+65%** |

### 6.7 SEPARABLE 分离近似（算法级优化）

**核心思想**：将 2D 双边滤波近似为水平 + 垂直两次 1D 滤波，复杂度从 $O(r^2)$ 降至 $O(r)$。r=5 时从 121 次降到 22 次邻域访问。

水平 pass 从 shared memory 沿行方向滤波，垂直 pass 沿列方向滤波。两个 pass 之间通过中间缓冲区（float 或 FP16）传递数据。

```cpp
// 水平 kernel: bilateral_filter_cuda.cu:281-293
#pragma unroll
for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
    float neighbor = smem[ty][lx + dx];
    float spatial_weight = d_spatial_lut[RADIUS * LUT_WIDTH + (dx + RADIUS)];
    float w = spatial_weight * color_weight;
    sum = fmaf(neighbor, w, sum);  // 显式 FMA
}
```

**中间缓冲区 SoA 布局**（Opt H）：水平输出改为 R|G|B 平面格式，垂直 kernel 读取单通道连续 float，合并效率从 ~33% 提升至 ~71%。

### 6.8 版本迭代汇总

> 以下 v1-v12 数据为早期开发阶段在 RTX 4060 (WSL2) 上的历史测量，vs OpenCV CPU 按当时基线（~57 ms）。最终行 SEPARABLE / SEP_FP16 为 Thor 实测。

| 版本 | 优化手段 | 4K RGB (ms) | 吞吐量 (MP/s) | vs OpenCV CPU |
|------|---------|---:|---:|---:|
| v1 | Naive global memory | 250 | 33 | 0.23x |
| v2 | Shared memory | 176 | 47 | 0.32x |
| v3 | + Spatial LUT | 140 | 59 | 0.40x |
| v4 | + `__expf` fast math | 55 | 150 | 1.03x |
| v5 | + Color LUT + unroll | 18 | 460 | 3.14x |
| v6 | + Template radius | 16.9 | 492 | 3.35x |
| v7 | + 持久缓冲 + LUT cache | 9.86 | 841 | 5.74x |
| v8 | + uint8 I/O kernels | 8.91 | 930 | 6.35x |
| v9 | + cudaHostRegister | 8.65 | 959 | 6.54x |
| v10 | + Block 16×16 | 8.64 | 960 | 6.55x |
| **v11** | **+ 单一色彩权重** | **7.45** | **1113** | **7.59x** |
| **v12** | **+ 圆形窗口 DCE** | **6.53** | **1271** | **8.67x** |
| — | SEPARABLE (Thor) | **3.06** | **2713** | **26.8x** |
| — | **SEP_FP16 (Thor)** | **3.01** | **2753** | **27.2x** |

> 从 naive 250 ms 到 TEMPLATE 6.53 ms 实现 **38x** 加速；Thor 上 SEP_FP16 3.01 ms 实现 **83x** 加速（vs naive）、**27.2x**（vs OpenCV CPU）。

---

## 7. 性能结果与质量验证

> 以下数据均在 **Jetson AGX Thor** 上实测（2026-03-16），统一内存架构，无 PCIe 传输瓶颈。

### 7.1 Thor — 4K RGB（3840×2160×3）

| 实现 | Time (ms) | Min (ms) | 吞吐量 (MP/s) | MAE | PSNR (dB) | vs OCV CPU | vs OCV CUDA |
|------|---:|---:|---:|---:|---:|---:|---:|
| **CUDA SEP_FP16** | **3.01** | **2.95** | **2753** | **0.46** | **48.39** | **27.2x** | **4.00x** |
| CUDA SEPARABLE | 3.06 | 2.99 | 2713 | 0.45 | 48.49 | 26.8x | 3.94x |
| CUDA TEMPLATE | 5.48 | 5.44 | 1515 | 0.60 | 48.28 | 15.0x | 2.23x |
| CUDA ADAPTIVE | 6.17 | 6.11 | 1344 | 0.40 | 49.42 | 13.3x | 1.94x |
| CUDA STANDARD | 9.41 | 9.26 | 882 | 0.48 | 48.61 | 8.70x | 1.29x |
| OpenCV CUDA | 12.07 | 11.79 | 687 | 0.00 | — | 6.79x | 1.00x |
| **OpenCV CPU** | **81.87** | **80.79** | **101** | — | — | **1.00x** | — |

### 7.2 Thor — 4K Grayscale（3840×2160×1）

| 实现 | Time (ms) | Min (ms) | 吞吐量 (MP/s) | MAE | PSNR (dB) | vs OCV CPU | vs OCV CUDA |
|------|---:|---:|---:|---:|---:|---:|---:|
| **CUDA SEP_FP16** | **1.32** | **1.27** | **6268** | **0.12** | **57.00** | **38.9x** | **5.88x** |
| CUDA SEPARABLE | 1.42 | 1.39 | 5832 | 0.15 | 56.18 | 36.4x | 5.53x |
| CUDA TEMPLATE | 3.52 | 3.46 | 2358 | 0.61 | 50.23 | 14.7x | 2.22x |
| CUDA STANDARD | 4.36 | 4.32 | 1901 | 0.61 | 50.23 | 11.9x | 1.76x |
| CUDA ADAPTIVE | 5.77 | 5.75 | 1438 | 0.61 | 50.23 | 8.94x | 1.34x |
| OpenCV CUDA | 7.78 | 7.44 | 1066 | 0.00 | — | 6.63x | 1.00x |
| **OpenCV CPU** | **51.60** | **50.57** | **161** | — | — | **1.00x** | — |

### 7.3 性能目标达成

| 目标 | 要求 | 最佳实测 | 模式 | 状态 |
|------|------|---:|------|:---:|
| 4K RGB @60fps | ≥498 MP/s | 2753 MP/s | SEP_FP16 | **5.5x 余量** |
| 4K Gray @60fps | ≥498 MP/s | 6268 MP/s | SEP_FP16 | **12.6x 余量** |
| vs OpenCV CPU | 显著加速 | **27.2x** (4K RGB) | SEP_FP16 | 达标 |
| vs OpenCV CUDA | > 1.0x | **4.00x** (4K RGB) | SEP_FP16 | 达标 |
| MAE | < 1.0 | 0.12~0.61 | 所有模式 | 达标 |
| PSNR | > 40 dB | 48.28~57.00 dB | 所有模式 | 达标 |

### 7.4 质量验证

| 模式 | MAE (RGB) | MAE (Gray) | PSNR (RGB) | PSNR (Gray) |
|------|---:|---:|---:|---:|
| STANDARD | 0.48 | 0.61 | 48.61 dB | 50.23 dB |
| TEMPLATE | 0.60 | 0.61 | 48.28 dB | 50.23 dB |
| SEPARABLE | 0.45 | 0.15 | 48.49 dB | 56.18 dB |
| SEP_FP16 | 0.46 | 0.12 | 48.39 dB | 57.00 dB |
| ADAPTIVE | 0.40 | 0.61 | 49.42 dB | 50.23 dB |

> SEPARABLE 在 Gray 下 MAE 仅 0.15，是所有模式中最接近 OpenCV 的结果。

---

## 8. Profiler 深度分析

### 8.1 nsys 时间线分析（Thor 平台）

#### TEMPLATE 模式

| 阶段 | 耗时 | 占比 |
|------|---:|---:|
| `k_bilateral_filter_rgb_template<5,u8,u8>` | 4.98 ms | **91%** |
| cudaMemcpy H2D (24.9 MB) | 0.20 ms | 3.6% |
| cudaMemcpy D2H (24.9 MB) | 0.21 ms | 3.8% |
| cudaLaunchKernel | 0.008 ms | <0.1% |

统一内存传输带宽：H2D **122 GB/s**，D2H **117 GB/s**，远高于 RTX 4060 的 PCIe（~8 GB/s）。

#### SEPARABLE 模式

| 阶段 | 耗时 | 占比 |
|------|---:|---:|
| Horizontal pass | 1.52 ms | **50.7%** |
| Vertical pass | 1.48 ms | **49.3%** |
| cudaMemcpy 合计 | 0.42 ms | ~12% |

**关键发现**：统一内存消除了 PCIe 传输瓶颈——Thor 上 memcpy 占比仅 ~5%（RTX 4060 上占 46-68%）。

### 8.2 ncu 硬件计数器 — TEMPLATE 模式

#### Speed-of-Light 概览

| 指标 | 值 | 说明 |
|------|---:|------|
| **SM Throughput** | **88.83%** | 接近计算峰值 |
| L1/TEX Throughput | 87.28% | Shared memory 密集 |
| L2 Throughput | 3.50% | 数据集 fit in L2 (32 MB) |
| DRAM Throughput | ~0% | 统一内存由 L2 服务 |
| **瓶颈诊断** | **Compute-bound** | SM 88.8% >> DRAM ~0% |

#### Occupancy 与寄存器

| 指标 | 值 |
|------|---:|
| 理论 Occupancy | 100% |
| **实测 Occupancy** | **97.76%** |
| 寄存器/线程 | 23 (sm_110) vs 64 (sm_89) |

> Blackwell 编译器将同一 kernel 的寄存器从 64 降至 23（-64%），occupancy 从 67% 提升至 97.8%。

#### 流水线利用率

| 流水线 | 利用率 | 说明 |
|--------|---:|------|
| **LSU** (Load/Store) | **43.84%** | Shared memory 读写 |
| **FMA** (浮点乘加) | **40.99%** | 权重计算 |
| ALU (整数) | 30.41% | 索引计算 |
| XU (超越函数) | 28.17% | 残余 exp 路径 |

> LSU 与 FMA **负载均衡**（43.8% vs 41.0%），计算与访存交织良好。

#### 缓存命中率

| 缓存层 | 命中率 |
|--------|---:|
| **Constant Cache** (LUT) | **99.99%** |
| Instruction Cache | 100.0% |
| L1/TEX | 68.53% |
| L2 | 73.31% |

### 8.3 ncu 驱动的关键优化（Thor 平台）

#### Opt G: Block 32×8 消除 Bank Conflict

**ncu 发现**：16×16 block 下 smem load 存在 50% bank conflict（2-way）。根因：warp 跨两行，行间 stride=27 mod 32=27，导致 11/16 bank 重叠。

**方案**：改为 32×8 block，warp 全在一行内（stride=1）。

| 指标 | 16×16 | 32×8 | 变化 |
|------|---:|---:|---:|
| Shared excessive wavefronts | 50% | **2.3%** | **-97.6%** |
| SM Throughput | 87.49% | 88.36% | +0.87pp |

#### Opt H: SoA 中间缓冲区改善合并访问

**ncu 发现**：SEPARABLE vertical kernel 全局内存合并效率仅 33%，68% sector 冗余。

| 指标 | AoS | SoA | 变化 |
|------|---:|---:|---:|
| Vertical uncoalesced | 68% | **29%** | **-39pp** |
| Horizontal uncoalesced | 69% | **47%** | **-22pp** |

#### Opt K: 激进 Launch Bounds 提升 Occupancy

**ncu 发现**：SEPARABLE RGB kernel 使用 61-63 regs，occupancy 仅 62%。

**方案**：`__launch_bounds__(256, 6)`，强制编译器将 regs 压到 ≤42（65536/256/6=42.67）。

| 指标 | 优化前 | 优化后 | 变化 |
|------|---:|---:|---:|
| Registers/thread | 63 | **40** | **-35%** |
| Achieved Occupancy | 62% | **97.5%** | **+35pp** |
| SM Throughput | 64.5% | **79.4%** | **+14.9pp** |
| 4K RGB min | 3.39 ms | **3.02 ms** | **-10.9%** |

> 这是单一改动（一个宏常量 `MIN_BLOCKS_PER_SM_SEP=6`）带来最大端到端收益的优化。关键：**零 spill**。

#### Opt N: FP16 中间缓冲区

**方案**：H↔V 中间缓冲区从 float 改为 `__half`，带宽减半。

| 指标 | FP32 | FP16 | 变化 |
|------|---:|---:|---:|
| V kernel LD sectors/req | 4.00 | **2.00** | **-50%** |
| H kernel ST sectors/req | 4.00 | **2.00** | **-50%** |
| 4K Gray min | 1.40 ms | **1.30 ms** | **-7.1%** |

### 8.4 Warp Stall 分析

| Stall 原因 | 比率 | 说明 |
|-----------|---:|------|
| **short_scoreboard** | **3.92** | Shared memory / L1 依赖 |
| not_selected | 3.21 | warp 调度竞争 |
| wait | 2.79 | 固定延迟依赖 |
| long_scoreboard | 0.66 | L2/DRAM 依赖 |
| math_pipe_throttle | 0.40 | 计算流水线满 |
| barrier | 0.33 | `__syncthreads()` |

### 8.5 Roofline 定位

```
SM Throughput (%)
  100 ┤
   89 ┤ ··[TEMPLATE]·················  (97.8% occ, 3.55 IPC)
      │
   67 ┤ ·········[SEP-V]             (63→97.5% occ after Opt K)
   65 ┤ ········[SEP-H]
      │
      └─────────────────────────────
        所有 kernel 均为 Compute-bound (SM >> Memory)
```

---

## 9. 失败实验与优化边界

### 9.1 Opt B: Warp Shuffle 水平 pass（-2.4x，已回退）

为 separable 灰度水平 kernel 实现 `__shfl_sync` 版本。

| 指标 | 基线 | Shuffle |
|------|---:|---:|
| 4K Gray | 2.07 ms | 5.02 ms |
| MAE | 0.15 | 0.50 |

**失败原因**：radius=5 时仅 10/32 lane 需要 halo 数据，if/else 分支导致严重 warp divergence。Shared memory 方案的加载延迟已被计算充分隐藏，shuffle 无法取得优势。

### 9.2 Opt C: Host 端 SoA 全链路（-36%，已回退）

将 RGB separable 改为 AoS→SoA 转换 + 3×灰度 separable + SoA→AoS 转换。

**失败原因**：(1) AoS↔SoA 转换引入 2 次额外全局内存遍历（47 MB）；(2) 8 个 kernel launch 替代原来 2 个；(3) 丧失三通道共享权重优势。

### 9.3 Opt D: cudaFuncCachePreferL1（-36%，已回退）

L1 偏好将 shared memory 从 64 KB 压缩到 32 KB/SM，导致 occupancy 下降，SEPARABLE 从 5.42 ms 恶化到 7.36 ms。

### 9.4 Opt M: Fused H+V 单 Kernel（-31~34%）

单 kernel 内三阶段（加载 2D halo → 水平滤波 → 垂直滤波），消除中间 buffer。

| 测试 | SEPARABLE | FUSED | 变化 |
|------|---:|---:|---:|
| 4K RGB | 3.02 ms | 3.96 ms | **-31%** |
| 1080p RGB | 0.77 ms | 1.03 ms | **-34%** |

**失败原因**：每 block 需处理 576 个水平滤波点（vs SEPARABLE 的 256），计算膨胀远超带宽节省。Thor 统一内存下中间 buffer 成本仅占 ~10%。

### 9.5 Opt N2: FP16 全量计算（-8.4%）

将 smem 和内循环累加全部改为 `__half`。

**失败原因**：(1) sm_110 的 2x FP16 优势仅在 `__half2` 打包操作上，标量 `__hfma` 与 FP32 `fmaf` 吞吐相同；(2) LUT 返回 float，每次迭代需 6+ 次 float↔half 转换。

### 9.6 实验总结

| 优化 | 状态 | 实际效果 | 教训 |
|------|:---:|---:|------|
| Warp Shuffle | 回退 | -2.4x | 需 halo 的滤波不适合 shuffle |
| Host SoA | 回退 | -36% | 格式转换开销 > 合并收益 |
| PreferL1 | 回退 | -36% | smem 容量比 L1 缓存更重要 |
| Fused H+V | 失败 | -31~34% | 统一内存下中间 buffer 成本低 |
| FP16 计算 | 失败 | -8.4% | 标量 half 无 2x，转换开销高 |
| fmaf 显式 | 保留 | 0% | nvcc -O3 已自动最优融合 |
| Strip Pipeline | 无收益 | 0~-8% | WSL2 阻止 copy+compute 并行 |

> 失败实验同样有价值——它们通过实测验证了优化边界，避免在无效方向继续投入。

---

## 10. 跨平台对比分析

### 10.1 Jetson AGX Thor 最终性能（4K RGB，实测 2026-03-16）

| 实现 | Avg (ms) | Min (ms) | 吞吐量 (MP/s) | vs OCV CPU | vs OCV CUDA |
|------|---:|---:|---:|---:|---:|
| **CUDA SEP_FP16** | **3.01** | **2.95** | **2753** | **27.2x** | **4.00x** |
| CUDA SEPARABLE | 3.06 | 2.99 | 2713 | 26.8x | 3.94x |
| CUDA TEMPLATE | 5.48 | 5.44 | 1515 | 15.0x | 2.23x |
| CUDA ADAPTIVE | 6.17 | 6.11 | 1344 | 13.3x | 1.94x |
| CUDA STANDARD | 9.41 | 9.26 | 882 | 8.70x | 1.29x |
| OpenCV CUDA | 12.07 | 11.79 | 687 | 6.79x | 1.00x |
| **OpenCV CPU** | **81.87** | **80.79** | **101** | **1.00x** | — |

### 10.2 跨平台 Kernel 耗时对比

| Kernel | Thor (ms) | RTX 4060 (ms) | cudaMemcpy |
|--------|---:|---:|---:|
| rgb_template<5> | 4.98 | 3.36 | — |
| horizontal_rgb<5> | 1.52 | 0.79 | — |
| vertical_rgb<5> | 1.48 | 0.72 | — |
| **H2D+D2H 合计** | **0.41** | **3.94** | **Thor 快 10x** |

### 10.3 编译器差异

| Kernel | sm_110 Regs | sm_89 Regs | sm_110 Occupancy | sm_89 Occupancy |
|--------|---:|---:|---:|---:|
| rgb_template | **23** | 64 | **100%** | 67% |
| gray_template | **21** | 63 | **100%** | ~67% |
| horizontal_rgb (Opt K) | **40** | 62 | **100%** | 66% |

### 10.4 平台特异性总结

| 因素 | 独显 (PCIe) | 统一内存 (Jetson) |
|------|:---:|:---:|
| 主瓶颈 | H2D/D2H (46-68%) | Kernel 计算 (90-95%) |
| 传输优化价值 | 极高 | 无意义 |
| Kernel 优化边际收益 | 中等（被传输稀释） | **极高**（直接反映端到端） |
| Fused kernel 价值 | 可能有效 | 无效（中间 buffer 成本低） |

---

## 11. 结论与展望

### 11.1 结论

本项目实现了从 naive CUDA kernel（250 ms）到高度优化的多模式双边滤波器，在 **Jetson AGX Thor** 上最终 SEP_FP16 模式 4K RGB 仅需 **3.01 ms**，总加速比 **83x**（vs naive）、**27.2x**（vs OpenCV CPU）、**4.00x**（vs OpenCV CUDA）。核心优化手段及其贡献：

| 排名 | 优化手段 | 加速贡献 | 代码位置 |
|:---:|---------|---:|------|
| 1 | Color Weight LUT | **~3x** | `d_color_lut[256]`, L52-53 |
| 2 | Shared Memory | **3-5x** | smem 协作加载, L96-107 |
| 3 | SEPARABLE 近似 | **O(r)** | H/V kernel, L241-522 |
| 4 | 持久 GPU 缓冲 | **+71%** | `g_bufs` 结构体, L1391-1467 |
| 5 | 圆形窗口 DCE | **+13~65%** | `spatial_weight==0` continue, L124 |
| 6 | RGB 单色权重 | **+16%** | 均值距离, L218-222 |
| 7 | launch_bounds(256,6) | **+10.9%** | `MIN_BLOCKS_PER_SM_SEP=6`, L41 |
| 8 | Template 展开 | **+7%** | `template<int RADIUS>`, L74 |

所有模式均满足：MAE < 1.0，PSNR > 48 dB，4K@60fps 吞吐余量 **5.5x** 以上。

### 11.2 展望

| 方向 | 可行性 | 预期收益 |
|------|:---:|:---:|
| `__half2` 向量化（打包两邻域像素） | 中 | FP16 吞吐翻倍 |
| Spatial LUT 搬到 smem（避免 constant cache 序列化） | 高 | 5-10% |
| L2 Cache 持久化（视频流场景） | 高 | 5-15% |
| 多 Stream Pipeline（原生 Linux，非 WSL2） | 高 | 传输与计算重叠 |
| Bilateral Grid（大半径 r≥10） | 低 | O(N) 复杂度 |

---

## 12. 附录：完整数据图表

### 表 A1: TEMPLATE 版本迭代（4K RGB, RTX 4060）

| 版本 | 优化手段 | Time (ms) | 吞吐量 (MP/s) | vs 上一版 | vs OCV CPU |
|------|---------|---:|---:|---:|---:|
| v1 | Naive | 250 | 33 | — | 0.23x |
| v2 | Shared memory | 176 | 47 | +42% | 0.32x |
| v3 | + Spatial LUT | 140 | 59 | +26% | 0.40x |
| v4 | + fast math | 55 | 150 | +154% | 1.03x |
| v5 | + Color LUT | 18 | 460 | +207% | 3.14x |
| v6 | + Template | 16.9 | 492 | +7% | 3.35x |
| v7 | + 持久缓冲 | 9.86 | 841 | +71% | 5.74x |
| v8 | + u8 I/O | 8.91 | 930 | +11% | 6.35x |
| v9 | + page-lock | 8.65 | 959 | +3% | 6.54x |
| v10 | + Block 16×16 | 8.64 | 960 | +1% | 6.55x |
| v11 | + 单色权重 | 7.45 | 1113 | +16% | 7.59x |
| v12 | + 圆形窗口 | 6.53 | 1271 | +13% | 8.67x |

### 表 A2: Thor 平台 4K RGB 全模式对比（实测 2026-03-16）

| 实现 | Avg (ms) | Min (ms) | 吞吐量 (MP/s) | MAE | vs OCV CPU | vs OCV CUDA |
|------|---:|---:|---:|---:|---:|---:|
| **SEP_FP16** | **3.01** | **2.95** | **2753** | **0.46** | **27.2x** | **4.00x** |
| SEPARABLE | 3.06 | 2.99 | 2713 | 0.45 | 26.8x | 3.94x |
| TEMPLATE | 5.48 | 5.44 | 1515 | 0.60 | 15.0x | 2.23x |
| ADAPTIVE | 6.17 | 6.11 | 1344 | 0.40 | 13.3x | 1.94x |
| STANDARD | 9.41 | 9.26 | 882 | 0.48 | 8.70x | 1.29x |
| OpenCV CUDA | 12.07 | 11.79 | 687 | 0.00 | 6.79x | 1.00x |
| OpenCV CPU | 81.87 | 80.79 | 101 | — | 1.00x | — |

### 表 A3: ncu 优化指标汇总

| 优化 | 指标 | Before | After | 变化 |
|------|------|---:|---:|---:|
| Opt G (32×8) | Smem bank conflict | 50% | 2.3% | -97.6% |
| Opt H (SoA) | Global uncoalesced (V) | 68% | 29% | -39pp |
| Opt H (SoA) | Global uncoalesced (H) | 69% | 47% | -22pp |
| Opt K (launch_bounds) | Registers/thread | 63 | 40 | -35% |
| Opt K (launch_bounds) | Achieved occupancy | 62% | 97.5% | +35pp |
| Opt K (launch_bounds) | SM throughput | 64.5% | 79.4% | +14.9pp |
| Opt N (FP16 temp) | V LD sectors/req | 4.00 | 2.00 | -50% |
| Opt N (FP16 temp) | H ST sectors/req | 4.00 | 2.00 | -50% |

### 表 A4: 编译期寄存器对比（sm_110 vs sm_89）

| Kernel | sm_110 Regs | sm_89 Regs | Spill | Occupancy (sm_110) |
|--------|---:|---:|:---:|---:|
| rgb_template<5,u8,u8> | 28 | 64 | 0 | 100% |
| gray_template<5,u8,u8> | 21 | 63 | 0 | 100% |
| horizontal_rgb<5,u8> | 40 | 62 | 0 | 100% |
| vertical_rgb<5,u8> | 40 | 62 | 0 | 100% |
| horizontal_gray<5,u8> | 33 | 35 | 0 | 100% |
