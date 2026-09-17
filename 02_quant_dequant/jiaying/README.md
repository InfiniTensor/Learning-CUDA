# MXFP8 / NVFP4 低精度软件模拟与反量化

2026 夏季训练营 CUDA 选题二。代码位于 `02_quant_dequant/jiaying/`，基于上游
`2026-summer-project` 分支。`jiaying` 是当前本地目录 ID，提交时可替换为训练营 ID。

本项目在 WSL2 Ubuntu 26.04 / RTX 4070 Laptop GPU 上完成编译与实测。
所有低精度数值转换均为软件实现，使用普通浮点、整数和 warp shuffle 指令。
不调用 FP8/FP4 Tensor Core、CUDA 原生 FP8/FP4 转换、Transformer Engine 或 cuBLAS。
默认生成 SM60、SM75、SM89 代码；FP16/BF16 输出也采用软件转换。

## 已实现

- FP32/FP16 行主序矩阵输入；MXFP8 E4M3 + E8M0；NVFP4 E2M1 + E4M3 + FP32。
- 标准块长 32/16，支持逐张量缩放和 2–1024 的二次幂块长实验。
- 最近偶数舍入和可复现随机舍入；保存真实 8 bit / 4 bit 数据及缩放因子。
- CUDA 量化、打包、解包和反量化；输出 FP16、BF16 或 FP32。
- 独立进程重新读取权重反量化；完整的头部、长度和非法编码校验。
- 单线程 CPU 基线、独立 NumPy 参考、全编码空间单测、111 个集成用例/后端。
- 三类分布 × 两种输入类型 × 两种格式 × 两种缩放模式，共 24 组实测日志。

总结和实测分析见 [REPORT.md](REPORT.md)，完整结果见
[reports/results/TABLE.md](reports/results/TABLE.md) 和 [summary.csv](reports/results/summary.csv)。

## 在当前 WSL 中运行

从 PowerShell 进入 WSL：

```powershell
wsl -d Ubuntu-26.04
```

以下命令在 WSL 中执行。当前机器已安装构建依赖，且已有 `build/low_precision`：

```bash
cd /mnt/d/Jiaying/Documents/ChatGPT/InfiniTensor/Learning-CUDA/02_quant_dequant/jiaying
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
bash scripts/build_wsl.sh

mkdir -p output
python3 scripts/generate.py output/input.bin --rows 1024 --cols 4096 \
  --distribution normal --dtype fp32 --seed 42

build/low_precision run --input output/input.bin --config configs/nvfp4.toml \
  --weights output/nvfp4.weights --output output/nvfp4.restored \
  --log output/nvfp4.json

build/low_precision decode --weights output/nvfp4.weights \
  --output output/reloaded_fp16.bin --output-type fp16
```

MXFP8：将配置换为 `configs/mxfp8.toml`。命令失败返回非零退出码。
`run` 每次自动核对 CPU/GPU 的全部权重、scale 字节、global scale 和输出字节。
`decode` 不需要原始输入或配置文件，可用 `--output-type` 覆盖文件中默认输出类型。

可选参数：`--backend cuda|cpu`（默认 cuda）、`--warmup 5`、`--iterations 50`；
`run` 另有 `--cpu-repeats 3`。输出目录需提前创建；输入、权重、配置、输出和日志路径必须不同。
普通 `decode` 不读取原始张量，因而不计算相对原始张量的误差；完整误差和性能日志由 `run` 生成。

## 其他环境构建

需要 CMake ≥ 3.20、支持 C++17 的编译器、CUDA Toolkit 和 NVIDIA 驱动；
测试和数据生成需要 Python 3 + NumPy。当前验证版本：CUDA 12.4.131、GCC 13.4.0。
安装 WSL 工具时使用 CUDA toolkit，运行时使用 Windows 驱动透传的 `/usr/lib/wsl/lib`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES='60;75;89'
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Toolkit 若已删除旧架构编译支持，需为 Pascal/Turing 使用 CUDA 12.x，
或通过 `CMAKE_CUDA_ARCHITECTURES` 选择该 Toolkit 支持的目标。
纯 CPU 构建用于调试参考，无需 CUDA：

```bash
cmake -S . -B build-cpu -DLP_ENABLE_CUDA=OFF -DCMAKE_CXX_COMPILER=g++-13
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
# 使用该程序时添加 --backend cpu
```

## 参数文件

```toml
format = "nvfp4"          # mxfp8 / nvfp4
block_size = 16           # 省略时由 format 决定，MXFP8=32，NVFP4=16
scale_mode = "block"      # block / tensor
output_type = "fp32"      # fp16 / bf16 / fp32
rounding = "nearest"      # nearest / stochastic
seed = 42                # uint64，随机舍入种子
target_gpu = "T4"         # 只作为日志说明，日志另记录真实 GPU
```

拒绝未知字段、重复字段和非法值。`#` 开始行内注释，双引号中的 `#` 保留。
这是简洁的 `key=value` 配置解析器，仅支持上述字段，不是完整 TOML 实现。
块沿每行最后一个维度划分，不跨行；尾块只统计有效元素。
逐张量模式在整个矩阵共享一个本地 scale。**非默认块长和逐张量模式属于对照实验扩展，
不宣称它们是原生 MXFP8/NVFP4 的标准硬件块布局。**

## 输入与输出的二进制协议

项目题目给出的是逻辑 header，未规定字符串长度和字节序；这里将其固定为以下版本化协议。
全部整数、IEEE 浮点和载荷采用 little-endian。读写工具为 `scripts/tensor_io.py`。

### 张量：固定 32 字节 header + 行主序载荷

| 偏移 | 字段 | 类型/约束 |
|---:|---|---|
| 0 | magic | 8 字节 `LPTENS1\0` |
| 8 | num_rows | uint64，值域为正 int64 |
| 16 | num_cols | uint64，值域为正 int64 |
| 24 | dtype | uint32：0=FP32，1=FP16，2=BF16（仅输出） |
| 28 | reserved | uint32=0 |
| 32 | values | rows×cols，FP32 每值 4 字节，FP16/BF16 每值 2 字节 |

可使用 `python3 scripts/generate.py output/input.bin --npy your_matrix.npy --dtype fp16`
导入已有二维 NumPy 矩阵。读取反量化结果：

```python
import sys
sys.path.insert(0, 'scripts')
from tensor_io import read_tensor
restored = read_tensor('output/nvfp4.restored')  # 返回二维 FP32 NumPy 数组
```

### 权重：固定 96 字节 header + packed data + scale bytes

| 偏移 | 字段 | 类型/含义 |
|---:|---|---|
| 0 | magic | 8 字节 `LPWGHT1\0` |
| 8 / 12 | version / header_bytes | uint32，1 / 96 |
| 16 / 24 | rows / cols | uint64，正 int64 范围 |
| 32 / 36 | format / block_size | uint32：0=MXFP8，1=NVFP4；块长 |
| 40 / 44 | scale_mode / output_type | uint32：0=block，1=tensor；同张量 dtype |
| 48 / 52 | rounding / input_type | uint32：0=nearest，1=stochastic；原输入 dtype |
| 56 | seed | uint64 |
| 64 / 72 | scale_count / packed_bytes | uint64 |
| 80 | global_scale | FP32；MXFP8 固定为 1 |
| 84 / 88 | flags / reserved | uint32=0 / uint64=0 |
| 96 | packed_data | MXFP8：rows×cols 字节；NVFP4：rows×ceil(cols/2) 字节 |
| 96+packed_bytes | scales | block：rows×ceil(cols/block_size) 字节；tensor：1 字节 |

NVFP4 的列 `2j` 位于低四位，列 `2j+1` 位于高四位。奇数列时，**每行**最后一个
高四位为零；不存在一个 4 bit 元素使用一个字节的展开存储。
例如 E2M1 的 `[1, -2]` 编码为 `[0x2, 0xc]`，打包字节为 `0xc2`。
此文件是本项目的紧凑权重格式，未采用 Blackwell GEMM 专用 scale swizzle/padding，
不能直接当作 Transformer Engine 的内部权重缓存。

## 数值约定

- E4M3 有限范围 ±448，E2M1 正数表为 `0, 0.5, 1, 1.5, 2, 3, 4, 6`。
- E8M0 编码 `e` 对应 `2^(e-127)`；0 编码不是零，255 为非法 NaN scale。
- 最近舍入是 ties-to-even；溢出低精度元素饱和到有限最大值。保留符号零和低精度次正规数。
- 随机舍入按相邻可表示值的距离选上下邻居，使用元素索引和 seed 生成随机数，
  与 CUDA 线程布局无关。缩放因子仍按最近偶数舍入，随机舍入只作用于元素。
- 全零 MX 块 scale=1；全零 NV 张量 global=1、local=0。
  极小 NV global 保底为 FP32 最小正次正规数，避免存储为零。
- 输入 NaN/Inf 明确报错；异常值分布指有限离群值。输出 FP16/BF16 遵循 IEEE 舍入，
  可能产生 Inf。此时日志记录 `nonfinite_output_count`，误差字段为 JSON `null`，不伪造有限误差。

## 复现实验

```bash
python3 scripts/benchmark.py                  # 24 组 1024×4096 实测，更新 reports/results
bash scripts/sanitize_wsl.sh                 # GPU 调试接口可用时执行四类检查
bash scripts/profile_wsl.sh                  # Nsight Systems（本机取得 CUDA API 时间线）
python3 scripts/package.py --output submission.zip
```

`reports/tests.txt` 保存 CTest 结果。当前 Windows/WSL 未启用 WDDM debugger interface，
Compute Sanitizer 的执行限制及原始错误见 `reports/sanitizer_mxfp8_memcheck.txt`。
常规 GPU 数值测试已经实际运行通过；不要将该限制误认为 sanitizer 已通过。

## 代码组织与提交

`include/codecs.hpp`：可在 CPU/GPU 共用的基础数值转换；`src/core.cpp`：单线程 CPU 基线；
`src/cuda_backend.cu`：GPU 归约、打包量化和解包反量化；`src/io.cpp`：格式校验与日志；
`src/main.cpp`：命令行；`tests/`：与产品实现分离的验收测试；`scripts/`：构建、数据和实验工具。

提交时把本目录放在个人 fork 的 `2026-summer-project` 分支下；保留源码、测试、配置、
脚本、报告和原始小型日志，不提交 `build*`、`output` 或大矩阵文件。
本地项目完成不代表已经向 GitHub 提交 PR。
