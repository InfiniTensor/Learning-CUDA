# 基于CUDA实现实时图像双边滤波

## 项目结构

```
Snowkyo16/
├── src/
│   ├── main.cu              # 主程序入口，版本调度器（NVIDIA A100 / Iluvatar BI100）
│   ├── main.maca            # 主程序入口，版本调度器（MetaX C500）
│   ├── main.mu              # 主程序入口，版本调度器（Moore Threads S5000）
│   ├── kernels.cu           # V1-V4 GPU kernel 及 wrapper 实现 （NVIDIA A100 / Iluvatar BI100）
│   ├── kernels.maca         # V5 GPU kernel 及 wrapper 实现（MetaX C500）
│   ├── kernels.mu           # V6 GPU kernel 及 wrapper 实现（Moore Threads S5000）
│   ├── bilateral_cpu.cpp    # V0 CPU 基线实现
│   ├── image_io.cpp         # 图像读写（基于 stb_image）
│   ├── params.cpp           # 滤波参数解析
│   └── benchmark.cpp        # 计时框架、性能汇总表
├── include/
│   ├── bilateral_filter.cuh # 各版本滤波函数声明
│   ├── image_io.h           # Image 结构体 + 读写接口
│   ├── params.h             # FilterParams 结构体
│   ├── benchmark.h          # 计时框架接口
│   ├── utils.cuh            # CUDA 错误检查宏
│   ├── stb_image.h          # 第三方：图像解码
│   └── stb_image_write.h    # 第三方：图像编码
├── scripts/
│   └── compare_opencv.py    # OpenCV 对比验证脚本
├── test_images/             # 测试图像
├── params.txt               # 默认滤波参数
├── Makefile                 # 编译构建
└── README.md
```

## 版本说明

| 版本 | 说明 | MODE 参数 |
|------|------|-----------|
| v0_cpu | CPU 基线实现 | `v0` |
| v1_naive | GPU Naive，一个线程一个像素 | `v1` |
| v2_smem | GPU Shared Memory Tiling | `v2` |
| v3_constmem | GPU Constant Memory LUT | `v3` |
| v4_stream | GPU Pinned Memory Stream Pipeline | `v4` |
| v5_metax | MetaX C500 MACA 移植（基于 V4 流水线） | `v5`（仅 MACA 平台） |
| v6_moore | Moore Threads S5000 MUSA 移植（基于 V4 流水线） | `v6`（仅 MUSA 平台） |
| all | 跑所有版本 + 性能对比表 | `all`（默认） |


## 编译与运行

### NVIDIA A100 平台

```bash
# 清理 
make clean

# 编译 
make build

# 运行
make run INPUT=test_images/yosemite.jpg
make run INPUT=test_images/yosemite.jpg MODE=v4

# 指定空闲 GPU
CUDA_VISIBLE_DEVICES=6 make run INPUT=test_images/yosemite.jpg
```

### Iluvatar BI-V100 平台

```bash
# 清理
PLATFORM=iluvatar make clean

# 编译
PLATFORM=iluvatar make build

# 运行
PLATFORM=iluvatar make run INPUT=test_images/yosemite.jpg
PLATFORM=iluvatar make run INPUT=test_images/yosemite.jpg MODE=v4
```

### MetaX C500 平台（MACA）

```bash
# 清理
PLATFORM=metax make clean

# 编译
PLATFORM=metax make build

# 运行
PLATFORM=metax make run INPUT=test_images/yosemite.jpg MODE=all
PLATFORM=metax make run INPUT=test_images/yosemite.jpg MODE=v5
```

### Moore Threads S5000 平台（MUSA）

```bash
# 清理 
PLATFORM=moore make clean

# 编译 
PLATFORM=moore make build

# 运行
PLATFORM=moore make run INPUT=test_images/yosemite.jpg MODE=all
PLATFORM=moore make run INPUT=test_images/yosemite.jpg MODE=v6
```

## OpenCV 对比验证

**通过标准:** MAE < 1，PSNR > 40 dB

```bash
# 验证各版本输出
python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v0_cpu.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v1_naive.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v2_smem.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v3_constmem.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v4_stream.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v5_metax.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v6_moore.png
```


## 性能分析 (nsys)
### nsys (时间线分析)
分析 API 调用耗时、内存传输、Kernel执行时间

```bash
# 采集profile
# 以 V4 为例（指定空闲 GPU）
CUDA_VISIBLE_DEVICES=6 nsys profile --trace=cuda -o output/v4_yosemite \
    make run INPUT=test_images/yosemite.jpg MODE=v4

# 查看统计报告
nsys stats output/v4_yosemite.nsys-rep
```

### ncu (Kernel 级分析)
分析单个 kernel 的计算吞吐、内存带宽、占用率、缓存命中率

```bash
# 分析 V3 kernel（跳过1次预热，分析1次调用）
CUDA_VISIBLE_DEVICES=6 sudo ncu \
    --kernel-name bilateral_filter_kernel_v3 \
    --launch-skip 1 --launch-count 1 \
    ./build/bilateral_filter test_images/yosemite.jpg config/default.txt output v3

# 分析 V4 的4个 strip kernel
CUDA_VISIBLE_DEVICES=6 sudo ncu \
    --kernel-name bilateral_filter_kernel_v3 \
    --launch-skip 1 --launch-count 4 \
    ./build/bilateral_filter test_images/yosemite.jpg config/default.txt output v4

# 导出报告文件（可用 Nsight Compute GUI 打开）
CUDA_VISIBLE_DEVICES=6 sudo ncu \
    --kernel-name bilateral_filter_kernel_v3 \
    --launch-skip 1 --launch-count 1 \
    -o output/ncu_v3_report \
    ./build/bilateral_filter test_images/yosemite.jpg config/default.txt output v3
```
