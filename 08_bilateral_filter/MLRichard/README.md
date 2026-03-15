# bilateral_filter

一个基于 CUDA 的实时双边滤波实验项目，默认适配 **英伟达（NVIDIA）GPU 环境**，支持灰度图与 RGB 图像，提供三阶段 CUDA kernel、CPU 参考实现、正确性测试与性能测试。项目优先面向训练营提供的英伟达环境，同时也可在标准 Linux / WSL + NVIDIA CUDA 环境中复用。

## 功能概览

- `Phase 1`：朴素 CUDA bilateral filter，作为正确性与性能基线
- `Phase 2`：Shared Memory 优化版本，减少邻域访问的全局内存开销
- `Phase 3`：自适应半径 + LUT 版本，结合 Sobel 梯度图进行区域自适应平滑
- `CPU` 参考实现：基于 OpenCV `cv::bilateralFilter`
- `测试`：
  - `test_correctness`：对比 OpenCV 输出，检查 MAE / PSNR
  - `test_performance`：统计 GPU / CPU 时间与吞吐量

## 目录结构

```text
bilateral_filter/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── params.txt
├── src/
│   ├── main.cu
│   ├── bilateral_filter.cu
│   ├── bilateral_filter.cuh
│   ├── io.cpp
│   ├── io.h
│   ├── reference.cpp
│   ├── reference.h
│   ├── metrics.cpp
│   └── metrics.h
├── test/
│   ├── test_correctness.cpp
│   └── test_performance.cpp
└── docs/
    └── plans/
```

## 环境要求

本项目默认面向英伟达平台，建议按以下方式准备环境：

- 如果你使用的是训练营提供的英伟达服务器，优先遵照课程中的英伟达环境文档完成基础配置。
- 如果你使用的是本地或其他英伟达环境，请确保至少具备：
  - 可用的 NVIDIA 驱动与 CUDA 环境
  - `C++17` 编译能力
  - `cmake`
  - `ninja` 或其他可用构建工具
  - OpenCV 开发环境
- 为了与当前 `CMakePresets.json` 保持一致，本文档下面给出一套推荐的 Conda 依赖配置，可直接用于训练营英伟达环境、普通 Linux 服务器或 WSL + NVIDIA CUDA 环境。

## NVIDIA 环境配置（推荐）

如果训练营服务器已经提供好基础环境，可直接跳到“激活环境”或“使用 CMake Presets”部分；以下步骤主要用于补齐本地或其他英伟达环境中的用户态依赖。

### 1. 初始化 Conda

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

### 2. 创建推荐环境

```bash
CONDA_NO_PLUGINS=true conda create -y \
  --solver classic \
  --override-channels \
  --strict-channel-priority \
  -n bilateral-cuda124 \
  -c nvidia/label/cuda-12.4.1 \
  -c conda-forge \
  python=3.10 cuda-toolkit opencv cmake ninja
```

### 3. 激活环境

```bash
conda activate bilateral-cuda124
```

### 4. 基础检查

```bash
nvcc --version
cmake --version
python -c "import cv2; print(cv2.__version__)"
nvidia-smi
```

预期：

- `nvcc` 可用
- `OpenCV` 可导入
- `nvidia-smi` 可看到 GPU

## 使用 CMake Presets

项目当前提供一组基于 Conda 环境的标准预设，可直接复用于英伟达训练营环境和其他标准 NVIDIA 环境：

- `conda-release`
- `conda-debug`
- `build-conda-release`
- `build-conda-debug`
- `build-test-correctness`
- `build-test-performance`
- `test-conda-correctness`
- `test-conda-performance`
- `test-conda-all`

### 配置 Release

```bash
cmake --preset conda-release
```

### 构建 Release

```bash
cmake --build --preset build-conda-release -j"$(nproc)"
```

### 仅构建正确性测试

```bash
cmake --build --preset build-test-correctness -j"$(nproc)"
```

### 仅构建性能测试

```bash
cmake --build --preset build-test-performance -j"$(nproc)"
```

### 使用 CTest 运行测试

运行正确性测试：

```bash
ctest --preset test-conda-correctness
```

运行性能 benchmark：

```bash
ctest --preset test-conda-performance
```

运行全部测试：

```bash
ctest --preset test-conda-all
```

默认构建目录：

- `build/conda-release`
- `build/conda-debug`

## 项目使用说明

### 主程序

```bash
./build/conda-release/bilateral_filter <input.raw> <output.raw> <params.txt> [kernel]
```

示例：

```bash
./build/conda-release/bilateral_filter test.raw out.raw params.txt shared
```

其中 `kernel` 可选：

- `naive`
- `shared`
- `adaptive`

默认值为 `shared`。

### 正确性测试

```bash
./build/conda-release/test_correctness
```

输出会包含：

- 图像尺寸与通道数
- `MAE`
- `PSNR`
- `PASS / FAIL`

### 性能测试

```bash
./build/conda-release/test_performance
```

输出会包含：

- GPU 时间
- CPU 时间
- 吞吐量（MPix/s）
- Speedup
- MAE / PSNR

## 输入 / 输出格式说明

### raw 文件格式

项目内部使用如下二进制格式：

```text
[4字节] width      (uint32_t, little-endian)
[4字节] height     (uint32_t, little-endian)
[4字节] channels   (uint32_t, little-endian)
[数据区] width * height * channels 字节，类型 uint8
```

说明：

- 灰度图：`channels = 1`
- RGB 图：`channels = 3`
- 像素按行主序存储
- RGB 使用交错布局：`R0,G0,B0,R1,G1,B1,...`

## 参数文件说明

默认参数文件 `params.txt` 内容如下：

```text
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
```

含义：

- `radius`：邻域半径
- `sigma_spatial`：空间高斯权重参数
- `sigma_color`：颜色权重参数

## 数据准备工具

项目附带两个辅助脚本，分别用于真实图片转换和基准输入生成。

### PNG / JPG 转 raw

将普通图片转换成项目使用的 raw 格式：

```bash
python tools/img2raw.py input.png output.raw
python tools/img2raw.py input.jpg output_gray.raw --gray
```

说明：

- 默认输出 RGB 三通道
- 传入 `--gray` 时输出灰度图
- 脚本优先使用 `Pillow`，缺失时会回退到 `OpenCV Python`

### 生成确定性测试输入

生成固定随机种子的 raw 文件，适合 correctness / benchmark / profiling：

```bash
python tools/generate_test_raw.py artifacts/profiling/inputs/test_4k.raw
python tools/generate_test_raw.py tmp/test_gray.raw --width 1024 --height 768 --channels 1 --seed 7
```

说明：

- 默认生成 `3840x2160 RGB` 测试图
- 输入内容可复现，便于不同机器之间对比吞吐与误差
- profiling 脚本会在缺少输入时自动调用这个工具

## 典型使用流程

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124

cmake --preset conda-release
cmake --build --preset build-conda-release -j"$(nproc)"

ctest --preset test-conda-correctness
ctest --preset test-conda-performance
```

## 验收对照

本项目按 [task.md](./task.md) 中提取的“方向 8”要求实现，当前可直接用下面的清单自查。

| 要求 | 当前实现 |
|------|----------|
| 灰度 / RGB 输入 | 支持，raw 格式 `channels=1/3` |
| `radius >= 5` | 支持，默认 `radius=5` |
| 可配置 `sigma_spatial` / `sigma_color` | 支持，来自 `params.txt` |
| 输出 raw 图像 | 支持，主程序写出 raw 文件 |
| CPU 参考实现 | 支持，基于 OpenCV `bilateralFilter` |
| 正确性指标 | `test_correctness` 检查 `MAE < 1.0` |
| 性能日志 | 主程序和 benchmark 都输出 `GPU Time / CPU Time / Throughput / Speedup` |
| 共享内存优化 | 支持，`shared` kernel |
| 自适应半径 | 支持，`adaptive` kernel |
| profiling | 支持 `ncu / nsys` 脚本链路 |

推荐在真实 GPU 环境中执行以下命令作为最终验收：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124
cmake --preset conda-release
cmake --build --preset build-conda-release -j"$(nproc)"
ctest --preset test-conda-all
```

如果 `ctest --preset test-conda-all` 全部通过，说明当前代码与 README 中记录的默认工作流一致。

## Profiling（Task 12 / NVIDIA 环境）

推荐直接使用统一脚本：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124
bash tools/profile_task12.sh
```

默认行为：

- 自动生成 `artifacts/profiling/inputs/test_4k.raw`
- 运行 `ncu` 采集 `naive / shared / adaptive`
- 运行 `nsys` 采集 `shared` 时间线
- 自动导出 `timeline.sqlite`
- 按 GPU 名称输出到 `artifacts/profiling/<gpu-slug>/`

常用变体：

```bash
# 仅跑 Nsight Systems
bash tools/profile_task12.sh --skip-ncu

# 使用已有 raw 输入
bash tools/profile_task12.sh --input /path/to/real_image.raw --skip-generate
```

说明：

- 在 **Windows + WSL + NVIDIA GPU** 下，脚本会自动补充 `/usr/lib/wsl/lib` 到 `LD_LIBRARY_PATH`
- 在 **Linux + NVIDIA GPU** 环境中，可直接复用同一脚本；若该路径不存在，也无需改项目源码
- 若 `ncu` 遇到 `ERR_NVGPUCTRPERM`，脚本会继续执行 `nsys`，并在输出目录生成 `ncu_permission_hint.txt`

## NVIDIA Linux 服务器兼容说明

项目默认适配英伟达平台。若你使用的是训练营提供的英伟达服务器，通常只需完成课程要求的基础环境配置即可直接复用本项目；若你使用的是其他 Linux 云服务器，也通常不需要修改项目源码。

推荐方式：

1. 服务器具备 NVIDIA GPU 与可用驱动
2. 安装 Miniconda / Conda
3. 使用与本地相同的 Conda 环境创建命令
4. 激活环境后直接使用相同的 `cmake --preset conda-release` 与 `cmake --build --preset build-conda-release`

说明：

- `CMakePresets.json` 主要依赖 `$env{CONDA_PREFIX}`，因此只要 Conda 环境结构一致，就不需要改路径。
- `LD_LIBRARY_PATH` 中加入 `/usr/lib/wsl/lib` 只是为了兼容 WSL；在普通 Linux 服务器上该路径通常不存在，也不要求修改项目源代码。
- 若服务器已有系统级 CUDA，也仍建议优先使用与项目一致的 Conda 环境，减少版本偏差。

## 课程提交说明

根据 [task.md](./task.md) 中提取的“项目代码要求”，最终交付时建议额外检查以下事项：

1. 提交仓库应位于 `Learning-CUDA` 的 `2025-winter-project` 分支体系下。
2. 目录布局应整理为 `/<选题>/<你的ID>/`。
3. 提交方式应为 `PR`，而不是只保留本地目录。
4. 最终交付版本应统一命名风格、补充关键注释，并根据课程要求确认是否需要移除测试代码。

如果课程原始文档在 `2026-03-03` 之后有更新，以最新的官方要求为准；本仓库中的 `task.md` 是对原始任务页的提取副本。

## 常见问题

### 1. `Failed to find nvcc`

说明当前环境中没有可用的 CUDA toolkit。

检查：

```bash
which nvcc
nvcc --version
```

处理：

- 确认已执行 `conda activate bilateral-cuda124`
- 确认环境中安装了 `cuda-toolkit`

### 2. VS Code / IntelliSense 提示 `#include` 错误

项目依赖：

- `compile_commands.json`
- `.vscode/c_cpp_properties.json`
- `.vscode/settings.json`

如果仍有报错：

```text
Developer: Reload Window
C/C++: Reset IntelliSense Database
```

并确认你已经先执行过：

```bash
cmake --preset conda-release
```

当前 IntelliSense 默认读取：`build/conda-release/compile_commands.json`。

### 3. 可以编译，但某些受限环境里 CUDA 运行失败

某些受限执行环境可以访问文件系统与编译器，但不允许真正访问 GPU compute。

表现通常为：

- `cudaGetDeviceCount` 失败
- `cudaMalloc` 报 `OS call failed or operation not supported on this OS`

这种情况不是项目代码错误，而是当前执行环境限制。

### 4. OpenCV 未找到

检查：

```bash
python -c "import cv2; print(cv2.__version__)"
ls "$CONDA_PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake"
```

如缺失，重新安装 `opencv` 包即可。

### 5. Nsight Compute 报 `ERR_NVGPUCTRPERM`

这是 **GPU Performance Counters 权限** 问题，不是项目代码错误。

项目已内置降级行为：

- `tools/profile_task12.sh` 会继续完成 `nsys` 采集
- 并在输出目录生成 `ncu_permission_hint.txt`

#### Windows + WSL + NVIDIA GPU 推荐处理步骤

根据 NVIDIA 官方说明，WSL 侧通常需要在 **Windows 主机** 开启计数器访问：

1. 以管理员身份打开 `NVIDIA Control Panel`
2. 勾选 `Desktop -> Enable Developer Settings`
3. 进入 `Developer -> Manage GPU Performance Counters`
4. 选择 `Allow access to the GPU performance counter to all users`
5. 重新打开 WSL 终端后再执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bilateral-cuda124
bash tools/profile_task12.sh
```

#### Linux / 云服务器 / A100 永久配置命令

```bash
sudo tee /etc/modprobe.d/nvidia-prof.conf >/dev/null <<'EOCONF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOCONF
sudo update-initramfs -u -k all
sudo reboot
```

重启后检查：

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

若输出为 `RmProfilingAdminOnly: 0`，说明普通用户可以运行 `ncu`。

官方说明：`https://developer.nvidia.com/ERR_NVGPUCTRPERM`

## 备注

- 当前推荐构建入口是 `CMakePresets.json`。
- 为了获得稳定体验，请始终在激活的 Conda 环境中执行 `cmake` 与 `ninja`。
- 如果需要更细粒度 profiling，可进一步结合 `ncu` / `nsys` 运行主程序分析 kernel 行为。
