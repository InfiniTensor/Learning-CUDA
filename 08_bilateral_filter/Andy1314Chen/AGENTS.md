# AGENTS.md — Bilateral Filter CUDA 项目

CUDA 双边滤波实现（InfiniTensor 训练营）。
目标 GPU：RTX 4060 (sm_89, Ada Lovelace)。环境：CUDA 13.1, WSL2。

## 构建

```bash
make                # 构建全部（需要 CUDA toolkit + OpenCV4）
make clean          # 清理构建产物
```

- **依赖**：CUDA toolkit (>= 11.0)、OpenCV4（`pkg-config --exists opencv4`）、C++17、GNU Make
- **编译选项**：`CXXFLAGS = -std=c++17 -O3 -Wall -Wextra`，`NVCCFLAGS = -O3 -arch=sm_89`
- **自动检测**：若 OpenCV 含 CUDA 模块则定义 `HAVE_OPENCV_CUDA`
- **无 CI/CD**；无单元测试框架。以 benchmark 验证正确性和性能（见下文）。

## 格式化

```bash
# .clang-format 位于 ../08_bilateral_filter/.clang-format（基于 LLVM）
clang-format -i src/*.cpp src/*.cu include/*.h include/*.cuh
```

核心规则：4 空格缩进、禁止 tab、100 列限宽、Attach 大括号风格、指针左对齐（`int* p`）、
include 自动分组排序（CUDA > C 系统头 > C++ 标准库 > 第三方 > 项目头文件）、`AlignConsecutiveMacros: true`。

## 运行 / 测试

### 单一模式运行
```bash
./bilateral_filter input.raw params.txt output.raw              # CPU
./bilateral_filter --cuda input.raw params.txt output.raw       # CUDA
./bilateral_filter --opencv input.raw params.txt output.raw     # OpenCV
```

### Benchmark（主要验证方式）
```bash
# CUDA vs OpenCV（推荐；跳过慢速 CPU）
./bilateral_filter --bench tests/test_data/input_4k.raw tests/test_data/params.txt

# 全量对比：CPU vs CUDA vs OpenCV
./bilateral_filter --compare-all tests/test_data/input_1080p.raw tests/test_data/params.txt
```

Benchmark 执行 5 次 warmup + 50 次计时。输出 mean/min/max/stddev (ms)、吞吐量 (MP/s)、MAE、PSNR。

### 指定 CUDA 实现模式（环境变量）
```bash
BILATERAL_MODE=0 ./bilateral_filter --bench ...   # STANDARD — 运行时半径
BILATERAL_MODE=1 ./bilateral_filter --bench ...   # TEMPLATE — 编译期半径（默认）
BILATERAL_MODE=2 ./bilateral_filter --bench ...   # SEPARABLE — 水平+垂直两趟，最快
BILATERAL_MODE=4 ./bilateral_filter --bench ...   # ADAPTIVE — 逐像素 Sobel 梯度自适应半径
BILATERAL_STRIP=4 ./bilateral_filter --bench ...  # 启用 strip pipeline（N>1 生效）
```

### 测试数据（`tests/test_data/`）
- `input_1080p.raw` / `input_4k.raw` — RGB（1920×1080 / 3840×2160）
- `input_1080p_gray.raw` / `input_4k_gray.raw` — 灰度
- `params.txt` — radius=5, sigma_spatial=3.0, sigma_color=30.0

### 验证标准
- **正确性**：MAE < 1.0、PSNR > 40 dB（以 OpenCV `bilateralFilter` 为基准）
- **性能**：记录耗时 (ms)、吞吐量 (MP/s)、相对 OpenCV CUDA 的加速比

### 性能分析
```bash
# password 123123
sudo /usr/local/cuda/bin/ncu --set full -o profile ./bilateral_filter --cuda input.raw params.txt output.raw
nsys profile -o timeline ./bilateral_filter --cuda input.raw params.txt output.raw
```

## 代码风格

### 文件结构
```
src/
  main.cpp                    # CLI 入口（--bench/--cuda/--opencv/--compare-all）
  bilateral_filter_cpu.cpp    # CPU 参考实现
  bilateral_filter_cuda.cu    # 全部 CUDA kernel（5 种模式）、LUT、strip pipeline
  bilateral_filter_opencv.cpp # OpenCV CPU + CUDA 封装、MAE/PSNR 计算
  image_io.cpp                # 二进制 raw 图像 I/O、参数解析
include/
  bilateral_filter.h          # CPU 滤波声明
  bilateral_filter_cuda.cuh   # CUDA 滤波声明
  bilateral_filter_opencv.h   # OpenCV + 质量指标声明
  image_io.h                  # ImageData/FilterParams 结构体、I/O 声明
```

### 命名规范
| 元素             | 规范             | 示例                           |
|------------------|------------------|--------------------------------|
| 文件             | snake_case       | `bilateral_filter_cuda.cu`     |
| 类/结构体        | PascalCase       | `ImageData`, `FilterParams`    |
| 函数             | snake_case       | `apply_bilateral_filter_cpu`   |
| CUDA kernel      | `k_` 前缀       | `k_bilateral_filter`           |
| 变量             | snake_case       | `sigma_spatial`                |
| 常量/宏          | UPPER_SNAKE_CASE | `MAX_RADIUS`, `CUDA_CHECK`     |

### include 顺序（由 `.clang-format` 强制）
```cpp
// 1. CUDA 头文件
#include <cuda_runtime.h>
// 2. C 系统头文件
#include <cstdio>
// 3. C++ 标准库
#include <vector>
// 4. 第三方库
#include <opencv2/opencv.hpp>
// 5. 项目头文件
#include "image_io.h"
```

### 格式化细则
- 4 空格缩进，禁止 tab，100 列限宽
- K&R 大括号风格（Attach），大括号不独占一行
- 头文件保护：`#ifndef FILE_NAME_H_` / `#define FILE_NAME_H_` / `#endif  // FILE_NAME_H_`
- 代码注释使用**英文**；`.md` 文档可用中文
- 公开 API 使用 Doxygen `@brief`/`@param` 注释

### 类型约定
- 像素值：`float`（GPU 计算）、`uint8_t`（存储/I/O）
- 尺寸：`int`（GPU kernel）或 `size_t`（host 端分配）
- GPU 指针：无别名时使用 `__restrict__`

### 错误处理
- I/O 函数返回 `bool`；失败时向 stderr 打印上下文信息
- 所有 CUDA API 调用用 `CUDA_CHECK()` 宏包裹（失败时 `exit(EXIT_FAILURE)`）
- `main()` 在参数非法或 I/O 失败时返回 1

```cpp
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)
```

### CUDA kernel 启动模式
```cpp
dim3 block(BLOCK_X, BLOCK_Y);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
k_bilateral_filter<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
```

## 文档

- **PROFILER.md** — 记录各优化版本的性能数据、roofline 分析
- **REPORT.md** — 优化实验记录（§9: Opt A–F）、profiler 分析（§10–12）
- 新增 benchmark 结果记录到 `PROFILER.md`；优化思路记录到 `REPORT.md`
- REPORT.md 很重要，需要认真整理，包括以下几点：
  - 详细阐述实现思路以及优化方法，具体包括但不限于优化历程和记录、开发中发现的问题等
  - 最终的性能指标和分析
  - 未来可继续提升的地方
  - 优化过程需要 ncu 和 nsys 使用和分析，指导优化方向
  - 实现思路应该从一个 navie cuda kernel 开始，然后逐渐迭代，逐步提升，记录每一步优化方法及分析思路（这里应结合 ncu 或 nsys 工具）
  - baseline 选择 opencv cpu 版本，opencv cuda 版本作为我们最初的一个目标，但最终我们的 CUDA 版本要超越 opencv cuda 版本
  - 优化过程中，需要记录每一步的优化结果，包括性能指标、roofline 分析等，指导优化方向，最好采用图表的形式记录下来

## 速查表

| 任务             | 命令                                                                              |
|------------------|-----------------------------------------------------------------------------------|
| 构建             | `make`                                                                            |
| 格式化           | `clang-format -i src/*.cpp src/*.cu include/*.h include/*.cuh`                    |
| Benchmark（快速）| `./bilateral_filter --bench tests/test_data/input_4k.raw tests/test_data/params.txt` |
| Benchmark（灰度）| `./bilateral_filter --bench tests/test_data/input_4k_gray.raw tests/test_data/params.txt` |
| 全量对比         | `./bilateral_filter --compare-all tests/test_data/input_1080p.raw tests/test_data/params.txt` |
| Separable 模式   | `BILATERAL_MODE=2 ./bilateral_filter --bench ...`                                 |
| 性能分析 (ncu)   | `ncu --set full -o prof ./bilateral_filter --cuda input.raw params.txt out.raw`   |
