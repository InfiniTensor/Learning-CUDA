# 实时图像双边滤波（CUDA）设计文档

## 文档信息

- 日期：2026-03-06
- 项目：实时图像双边滤波（CUDA）
- 目标平台：NVIDIA RTX 4070（本地开发）、NVIDIA A100（性能测试）

---

## 一、总体设计

### 架构总览

```
输入 (raw / PNG / JPG)
    │
    ▼
[CPU] 读取图像 → 转换为浮点 → 传输到 GPU
    │
    ▼
[GPU] Phase 1: 朴素 Kernel（基线）
[GPU] Phase 2: Shared Memory + 预计算权重表
[GPU] Phase 3: 自适应半径 + 查找表替代 exp()
    │
    ▼
[CPU] 结果回收 → 与 OpenCV 参考结果对比（MAE / PSNR）
    │
    ▼
输出 (raw) + 性能日志
```

### 核心约束

- 输入/输出格式：二进制 raw（宽、高、通道数 + 行主序像素数据）
- 支持灰度图（1 通道）和 RGB 图（3 通道），像素值 0-255
- 窗口半径 r 支持至少 5（11×11 窗口）
- 正确性：MAE < 1.0（对比 OpenCV cv::bilateralFilter）
- 性能目标：在 A100 上处理 4K RGB 图像达到实时水平（> 60fps）

---

## 二、项目结构

```
bilateral_filter/
├── CMakeLists.txt
├── params.txt                   # 默认参数文件
├── README.md
├── src/
│   ├── main.cu                  # 程序入口，参数解析，流程控制
│   ├── bilateral_filter.cu      # CUDA kernel（三个阶段）
│   ├── bilateral_filter.cuh     # kernel 声明和参数结构体
│   ├── io.cpp                   # raw 文件读写，PNG/JPG 转 raw
│   ├── io.h
│   ├── reference.cpp            # CPU 参考实现（调用 OpenCV）
│   ├── reference.h
│   ├── metrics.cpp              # MAE / PSNR 计算
│   └── metrics.h
├── test/
│   ├── test_correctness.cpp     # 自动生成测试图，验证 MAE < 1
│   └── test_performance.cpp     # benchmark，输出吞吐量
└── docs/
    ├── plans/
    │   └── 2026-03-06-bilateral-filter-design.md
    └── report.md                # 总结报告（最终交付）
```

---

## 三、CUDA Kernel 设计

### Phase 1 — 朴素 Kernel（基线）

- 1 线程 : 1 输出像素
- block size：16×16（profile 后可调整为 32×32）
- 每次直接从 Global Memory 读取邻域像素
- 用途：正确性基线，提供性能对比数据

```
kernel_naive<<<grid, block>>>(input, output, width, height, channels,
                               radius, sigma_spatial, sigma_color)
```

### Phase 2 — Shared Memory Kernel

- Tile size：`BLOCK_SIZE × BLOCK_SIZE`
- Halo 区域：tile 周围额外加载 r 圈像素（halo），大小为 `(BLOCK_SIZE + 2r)²`
- 所有线程协作将 halo 区域从 Global Memory 加载到 Shared Memory
- 空间高斯权重表：预计算 `(2r+1)²` 个值，存入 Constant Memory（最大 r=15）
- 颜色权重：运行时计算 `exp(-Δcolor² / (2σc²))`

```
__constant__ float c_spatial_weight[(2*MAX_RADIUS+1)*(2*MAX_RADIUS+1)];

kernel_shared<<<grid, block, shared_mem_size>>>(...)
```

### Phase 3 — 自适应半径 Kernel

- 步骤 1：跑 Sobel kernel，计算每像素梯度幅值图
- 步骤 2：按梯度阈值分配半径
  - 梯度 > T（边缘区域）→ 小半径（r=3），保边
  - 梯度 ≤ T（平坦区域）→ 大半径（r=7），强平滑
- 步骤 3：颜色权重用 256 级查找表替代 `exp()` 计算
  - 查找表预计算 `lut[i] = exp(-i² / (2σc²))`，i ∈ [0, 255]
  - 存入 Constant Memory 或 Texture Memory

---

## 四、内存布局

- GPU 上图像以 **行主序 float 数组** 存储
- RGB 图：交错存储 `[R0,G0,B0, R1,G1,B1, ...]`
- 灰度图：`[V0, V1, V2, ...]`
- 颜色距离：RGB 用三通道欧氏距离 `√(ΔR²+ΔG²+ΔB²)`，灰度用绝对差
- 内存分配：初版用 `cudaMalloc + cudaMemcpy`，后期可升级为 `cudaMallocPitch` 做内存对齐优化

---

## 五、参数文件格式

```text
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

---

## 六、输入/输出文件格式

### raw 文件格式

```
[4字节] width  (uint32_t, little-endian)
[4字节] height (uint32_t, little-endian)
[4字节] channels (uint32_t, little-endian)
[width × height × channels 字节] 像素数据 (uint8_t, 行主序)
```

---

## 七、正确性验证

- 自动生成随机噪声图（512×512 RGB 和灰度）用于快速 CI 测试
- 调用 `cv::bilateralFilter` 作为参考，计算 MAE 和 PSNR
- 断言 MAE < 1.0
- 支持 Set14 / BSDS100 真实图像批量验证
- DIV2K 验证集（100 张）用于 4K 性能 benchmark

---

## 八、性能日志格式

```
Platform     : NVIDIA A100
Image        : 3840x2160 RGB
Kernel       : Phase2_SharedMemory
Radius       : 5, sigma_s=3.0, sigma_c=30.0
GPU Time     : 12.34 ms
Throughput   : 673.2 MPix/s
CPU Time     : 1823.5 ms (OpenCV)
Speedup      : 147.8x
MAE          : 0.42
PSNR         : 51.3 dB
```

---

## 九、测试数据集

| 用途 | 数据集 | 规模 |
|------|--------|------|
| 快速 CI 验证 | 程序自动生成随机图 | 512×512 |
| 正确性对比 | Set14 / BSDS100 | < 50 MB |
| 性能 Benchmark | DIV2K 验证集 | ~400 MB |

---

## 十、Profiling 计划

- 工具：ncu（Nsight Compute）+ nsys（Nsight Systems）
- 平台：RTX 4070（本地）和 A100（远程服务器）对比
- 分析重点：
  - Global Memory 带宽利用率
  - Shared Memory bank conflict
  - Occupancy 和 warp 利用率
  - 各阶段 kernel 耗时对比
- 报告中包含 ncu/nsys 截图和分析

---

## 十一、实现阶段规划

| 阶段 | 内容 | 完成标志 |
|------|------|---------|
| Phase 1 | 朴素 Kernel + IO + 参考实现 + MAE 验证 | MAE < 1.0，程序可跑通 |
| Phase 2 | Shared Memory + 预计算权重 + benchmark | 性能明显优于 Phase 1 |
| Phase 3 | 自适应半径 + 查找表 + ncu/nsys profiling | 完整性能报告 |
