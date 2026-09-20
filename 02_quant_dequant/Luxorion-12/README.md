# Luxorion
本项目从 CPU reference 开始，逐步实现 MXFP8、NVFP4、CUDA kernel、文件 IO、指标统计、正确性测试以及 CUDA 性能优化。
当前已完成：E2M1 / E4M3 / E8M0 编解码、FP4 紧凑打包、MXFP8 / NVFP4 的 CPU 与 CUDA quantize/dequantize、误差与压缩率计算、固定 20 字节 header 的 FP32/FP16 `tensor.bin` 读写、量化参数解析、原生 FP16 GPU 输入、tensor/block scaling、nearest/stochastic rounding、版本化低精度权重文件、FP16/BF16/FP32 反量化输出，以及 v0 到 v8 的逐项 CUDA 优化。
当前默认正式入口为：
```text
gpu_baseline/tools/luxorion_cli.cu
```
其实现与：
```text
optimization/v8_ilp_unroll/luxorion_cli.cu
```
保持一致。
`luxorion_cli` 读取：
```text
tensor.bin + quant.cfg
```
并输出：
```text
weights.luxq
restored.tensor.bin
report.json
```
## 目录结构
```text
Luxorion-12/
├─ cpu_reference/        CPU参考实现、文件IO、配置解析、指标和单元测试
├─ gpu_baseline/         CUDA baseline、benchmark、CUDA测试和正式v8 CLI
├─ input/                冻结测试输入、manifest和量化配置
├─ optimization/         v0-v8优化历史、实验记录和性能对照
├─ .gitignore
├─ CMakeLists.txt        工程构建入口
├─ README.md             使用和代码阅读说明
└─ REPORT.md             项目原理、实现、优化与实验报告
```
以下目录由程序运行或构建时生成，不提交仓库：
```text
build/
output/
```
## 模块职责与主流程
建议先理解各模块负责什么，再进入函数内部查看具体实现。
| 模块 | 负责解决的问题 | 输入与输出 | 主流程位置 |
| --- | --- | --- | --- |
| `main.cpp` | 将 CPU reference 组织成可运行的教学演示 | 固定样例与输出目录 → 日志/CSV | 准备输入、调用算法、展示结果 |
| `formats.cpp` | 单值低精度格式编解码及 E2M1 打包 | 浮点值/编码 → 编码/浮点值/packed byte | 被 MXFP8/NVFP4 量化和恢复调用 |
| `mxfp8_cpu.cpp` | 模拟 MXFP8 的共享 E8M0 scale + E4M3 元素 | FP32 数组 → `MXFP8Result` → 恢复数组 | CPU MXFP8 reference |
| `nvfp4_cpu.cpp` | 模拟 NVFP4 的 global/local scale 和 E2M1 packed storage | FP32 数组 → `NVFP4Result` → 恢复数组 | CPU NVFP4 reference |
| `metrics.cpp` | 计算数值误差和存储收益 | 原始/恢复数组或字节数 → 指标 | 正确性与实验统计 |
| `io.cpp` | 读取/写入 FP32/FP16 `tensor.bin` | 二进制文件 ↔ 张量数据 | CPU reference 与正式 CLI 共用 |
| `quant_config.cpp` | 解析并校验量化参数 | `quant.cfg` → `QuantConfig` | 正式 CLI 配置入口 |
CPU reference 的基本计算流程为：
```text
创建或读取输入
    ↓
MXFP8 / NVFP4 quantize
    ↓
保存低精度编码与 scale
    ↓
对应 dequantize
    ↓
恢复值
    ↓
与原始值计算 MaxAbs / MAE / MSE
    ↓
计算量化负载和压缩率
```
头文件负责声明接口和数据结构，源文件负责具体实现，测试文件负责验证格式规则、边界条件和错误输入。
## CPU reference 阅读顺序
CPU reference 的主要作用是定义项目的数值语义，并作为 CUDA 实现的独立 oracle。它同时包含文件 IO、量化配置解析和指标计算，但不包含 CUDA `.cu` 实现。
推荐按照以下顺序阅读：
1. **整体流程**：`cpu_reference/src/main.cpp` 的 `main`，理解输入、量化、反量化、统计与输出之间的关系。
2. **E2M1 单值编码**：`cpu_reference/src/nvfp4_cpu.cpp` 中的 `kMagnitude`、`encode_e2m1`、`decode_e2m1`。
3. **E4M3 / E8M0 与打包**：`cpu_reference/src/formats.cpp`，查看 E4M3、E8M0 以及 `pack_e2m1` / `unpack_e2m1`。
4. **MXFP8 分块**：先看 `cpu_reference/include/mxfp8.h`，再看 `cpu_reference/src/mxfp8_cpu.cpp`。
5. **NVFP4 两级 scale**：先看 `cpu_reference/include/nvfp4.h`，再看 `cpu_reference/src/nvfp4_cpu.cpp`。
6. **指标计算**：`cpu_reference/include/metrics.h` 与 `cpu_reference/src/metrics.cpp`。
7. **文件与配置**：`cpu_reference/src/io.cpp` 与 `cpu_reference/src/quant_config.cpp`。
8. **测试**：`cpu_reference/tests/` 中的格式、随机、边界、文件和非法输入测试。
源码中的中文注释给出了公式、关键变量、位布局和具体数值示例。第一次阅读时可以暂时跳过日志打印、文件流等样板代码。
## 运行 CPU 演示
构建后，在 `Luxorion-12` 目录运行：
```powershell
.\build\release\cpu_reference.exe
.\build\release\cpu_reference.exe output\cpu_demo
```
程序会打印各元素的 scale、编码、恢复结果及汇总指标，并生成：
- `input.csv`：两个固定样例的原始输入。
- `mxfp8.csv`：MXFP8 元素、E8M0 scale、E4M3 编码、恢复值和误差。
- `nvfp4.csv`：NVFP4 元素、global/local scale、E2M1 编码和误差。
- `nvfp4_packed.csv`：实际 packed byte 及高低四位布局和尾部填充。
- `metrics.csv`：MaxAbs、MAE、MSE、原始字节数、量化负载字节数和压缩率。
MXFP8 示例展示块级 E8M0 scale 和 E4M3 饱和/舍入行为。
NVFP4 示例展示 global/local 两级 scale、E2M1 编码以及两个 4-bit 元素打包到一个字节的过程。
这些 CSV 主要用于人工阅读和核对，不是正式输入/权重文件协议。
正式二进制协议为：
```text
tensor.bin
weights.luxq
```
## 测试数据与验证
`cpu_reference/tests/` 包含格式编解码、手工样例、随机输入、边界条件、文件协议、配置解析和错误输入测试。
CPU 测试可以通过：
```powershell
ctest --test-dir build\release --output-on-failure
```
启用 CUDA 构建后，测试还会加入 GPU 与 CPU oracle 对照以及正式 CLI 流程测试。
当前完整 CUDA 构建包含 13 项 CTest。
## 正式量化与反量化
完整 GPU 量化—反量化闭环：
```powershell
.\build\cuda\Release\luxorion_cli.exe run `
  input\benchmark\normal_fp16_1024x1024_seed1234.tensor.bin `
  input\quant.cfg `
  output\run_example
```
输出：
```text
output/run_example/
├─ weights.luxq
├─ restored.tensor.bin
└─ report.json
```
其中：
- `weights.luxq`：版本化低精度权重文件；
- `restored.tensor.bin`：FP16/BF16/FP32 反量化结果；
- `report.json`：量化/反量化时间、p95、有效带宽、误差和压缩率等指标。
独立反量化不需要重新读取原始输入：
```powershell
.\build\cuda\Release\luxorion_cli.exe dequantize `
  output\run_example\weights.luxq `
  output\run_example\restored_again.tensor.bin
```
`run` 和 `dequantize` 都可以在命令末尾指定 warmup 和 repeats。
默认：
```text
warmup = 10
repeats = 100
```
stochastic rounding 使用 `quant.cfg` 中可选的 `seed`，未填写时默认使用 `1234`。
## CUDA baseline 与正式 v8
`gpu_baseline/` 中同时保留 CUDA baseline 和当前正式 CLI。
### CUDA baseline
```text
gpu_baseline/src/mxfp8.cu
gpu_baseline/src/nvfp4.cu
```
这里保留以正确性和可读性为主的基础 CUDA 实现，用于 CPU/GPU 对照和基础性能 benchmark。
MXFP8 baseline：
```text
32个连续元素
    ↓
一个32-thread CUDA block
    ↓
求 block amax
    ↓
选择 E8M0 scale
    ↓
每线程编码一个 E4M3 元素
```
NVFP4 baseline：
```text
整个张量求 global amax
    ↓
计算 FP32 global scale
    ↓
每16元素一个 CUDA block
    ↓
计算 E4M3 local scale
    ↓
编码 E2M1
    ↓
每两个元素打包成1 byte
```
`gpu_baseline/tests/test_mxfp8_cuda.cpp` 和 `gpu_baseline/tests/test_nvfp4_cuda.cpp` 会逐层比较：
```text
CPU oracle
    ↕
CUDA baseline

scale
编码
packed data
恢复值
signed zero
```
### 正式优化版本
当前正式 CLI：
```text
gpu_baseline/tools/luxorion_cli.cu
```
已经更新到最终 v8 实现。
v0-v8 的完整优化过程保存在：
```text
optimization/
```
最终实际晋级链为：
```text
v0 → v1 → v3 → v6 → v7 → v8
```
未晋级版本同样保留，用于记录无收益或回退的优化实验。
## Benchmark 输入
冻结性能输入位于：
```text
input/benchmark/
```
当前共包含 12 个输入文件：
```text
uniform
normal
outliers
zero-heavy
multiscale
tail-normal
```
每一类同时提供：
```text
FP32
FP16
```
即：
```text
6种数据分布 × 2种dtype = 12个文件
```
其中 `tail-normal` 使用非整齐元素数量，用于验证不能被 16/32 整除时的尾块处理。
`manifest.json` 记录：
```text
shape
dtype
seed
distribution
block layout
```
固定输入由：
```powershell
.\build\cuda\Release\generate_gpu_baseline_inputs.exe
```
生成并通过正式 `tensor.bin` 读取器回读校验。
正常 benchmark 不会自动重新生成这些文件。
## Benchmark 协议
运行：
```powershell
.\build\cuda\Release\gpu_baseline_benchmark.exe
```
默认协议：
```text
warmup = 10
repeats = 100
```
输出：
```text
median
nearest-rank p95
effective GB/s
MaxAbs
MAE
MSE
compression ratio
device pipeline time
end-to-end time
```
量化、反量化和 device-resident pipeline 使用 CUDA Event 计时。
端到端时间使用 host `steady_clock`，包含：
```text
H2D
+
quantize
+
dequantize
+
D2H
```
不包含：
```text
文件IO
显存分配
CPU oracle验证
```
结果默认写入：
```text
output/gpu_baseline/
├─ baseline_results.csv
└─ baseline_results.json
```
`output/` 为运行时目录，不提交 Git 仓库。
## CUDA 构建
本机测试环境为：
```text
NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability 8.6
CUDA Toolkit 12.8
```
对应 PowerShell 构建命令：
```powershell
cmake -S . -B build\cuda `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DLUXORION_ENABLE_CUDA=ON `
  -DLUXORION_BUILD_HISTORY=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build\cuda --config Release

ctest --test-dir build\cuda -C Release --output-on-failure
```
默认：
```text
LUXORION_BUILD_HISTORY=OFF
```
正常情况下只构建正式程序、benchmark 和测试，不构建全部 v0-v8 历史可执行文件。
如果需要复现优化历史：
```powershell
cmake -S . -B build\cuda `
  -DLUXORION_ENABLE_CUDA=ON `
  -DLUXORION_BUILD_HISTORY=ON
```
历史源码始终保留，`OFF` 只表示不参与默认构建。

## 按 GPU 整理的最终文件

`gpu_final/nvidia_rtx3050/` 保存 NVIDIA RTX 3050 的最终源码、可执行文件与校正后的结果；`gpu_final/metax_c500/` 保存沐曦 C500 的最终 v7 源码、MXMACA 兼容层、构建/复测脚本与实测数据。所有命令均从项目根目录使用相对路径执行。
其他 GPU 可以将：
```text
CMAKE_CUDA_ARCHITECTURES=86
```
替换为对应计算能力；也可以在全新的构建目录中省略该参数，让 CMake 使用 `native`。
普通 CPU 构建默认不启用 CUDA，不需要 CUDA 工具链。
