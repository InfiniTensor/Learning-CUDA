# MXFP8 / NVFP4 软件模拟与反量化（CUDA）

本项目在普通 CUDA core 上实现 MXFP8 和 NVFP4 的编码、缩放、真实位宽打包、
解包与反量化，不调用原生 FP8/FP4 类型、Tensor Core、Transformer Engine 或
cuDNN。支持 FP32/FP16 输入，FP16/BF16/FP32 输出，per-tensor/per-block scale，
round-to-nearest-even 和确定性 stochastic rounding。

## 构建与测试

需要 CUDA Toolkit 11+、CMake 3.20+、C++17 和 NVIDIA GPU。默认 Makefile 目标
为 T4（SM75）：

```bash
make CUDA_HOME=/usr/local/cuda CUDA_ARCH=75 -j
make test CUDA_HOME=/usr/local/cuda CUDA_ARCH=75
```

也可以直接用 CMake；应把架构号换成实际 GPU，例如 RTX 50 系列为 120：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

自测覆盖 E4M3/E8M0/E2M1 精确值、MXFP8/NVFP4、tensor/block scale、随机均匀、
正态和含异常值分布。CLI 测试还覆盖 FP16 输入、FP16/BF16 输出、奇数元素个数的
4-bit 尾字节以及输出文件布局。

## 硬件/库对照路径（可选，加分项）

默认关闭，关闭时构建产物与基础版完全一致。在支持 FP8/FP4 的 GPU（sm_89+ 的
cvt 指令；GEMM 需要 Blackwell 低精度 Tensor Core）上可打开：

```bash
make BUILD_DIR=build-hw WITH_HW_PATH=1 \
  CUDA_HOME=~/cuda-root/usr/local/cuda-12.9 CUDA_ARCH=120 -j
```

cuBLASLt（GEMM 路径）通过 `-DCUBLAS_ROOT=...` 指定（pip 的 nvidia-cublas 包
或完整 CUDA 安装均可）；找不到时只编译 cvt 指令路径。打开后配置文件中可设置
`hardware_path = "off|cvt|gemm"`（默认 `off`，不影响软件路径输出）：

- `cvt`：用 SM 原生 `cvt.rn.satfinite.e4m3` 指令做 MXFP8 元素编解码，与软件
  位操作路径逐字节对比并计时，结果写入 `hw_path.json`；
- `gemm`：推理场景 Y = X·Wᵀ 对照——软件「反量化到 FP16 + cublasGemmEx」
  vs cuBLASLt 原生 FP8（VEC32_UE8M0 块缩放）/FP4（VEC16_UE4M3 块缩放）
  GEMM，结果写入 `hw_gemm.json`。

一键基准（1024×1024 与 4096×4096）：

```bash
./scripts/run_hw_benchmarks.sh
```

## Hadamard 旋转融合（可选，方向三联动）

默认关闭，关闭时构建产物与之前完全一致。打开后相对路径直接编译选题三目录下的
`../../03_hadamard_tc/yangyeyuan/src/hadamard.cu`（不复制源码），量化前对每个 128 列 chunk
做归一化 FWHT 旋转，反量化后再变换一次作为逆旋转（归一化 Hadamard 正交且
对称）：

```bash
make BUILD_DIR=build-had WITH_HADAMARD=1 \
  CUDA_HOME=~/cuda-root/usr/local/cuda-12.9 CUDA_ARCH=120 -j
```

打开后配置文件中可设置 `hadamard = "off|chunk128"`（默认 `off`，不影响软件
路径输出）。chunk128 要求输入列数能被 128 整除且 `output_type = "fp16"`；
旋转计入量化计时、逆旋转计入反量化计时。未用 `WITH_HADAMARD=1` 构建时设置
`hadamard = "chunk128"` 会直接报错而不是静默忽略。

## 输入与运行

张量文件由 ASCII header 和紧随其后的原始 little-endian 数据组成：

```text
[header]
num_rows: 1024
num_cols: 1024
dtype: fp32

[data]
<1024*1024 个 float 的二进制字节>
```

量化配置示例见 `configs/`。运行命令：

```bash
./build/low_precision_cli \
  --input input.bin --config configs/nvfp4.toml \
  --output-dir results/my_run --warmup 20 --iters 200
```

无 NumPy 依赖的测试数据生成器可以直接生成三类输入：

```bash
python3 tests/generate_inputs.py --output examples --rows 1024 --cols 1024
```

输出目录包含：

- `quantized.lpq`：88-byte versioned header，随后为 packed data 和 1-byte scale
  数组。NVFP4 中偶数下标在低 nibble，奇数下标在高 nibble；奇数长度最后的高
  nibble 为零。
- `dequantized.bin`：无 header、row-major 原始数组，元素类型由 `output_type`
  指定。
- `metrics.json`：误差、压缩率、CUDA-event kernel 时间和按最少输入/输出流量
  计算的有效带宽。

输入中的 NaN/Inf 会被拒绝，避免误差统计失去意义。最后一个不足 block_size 的
分组按实际有效元素计算 scale。`scale_mode=tensor` 把整个线性张量当成一个分组；
canonical MXFP8/NVFP4 则分别使用 32/16 元素的 block mode。

## 一键基准与 profiling

```bash
./scripts/run_benchmarks.sh
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

基准结果写入 `results/*/metrics.json`。Nsight 脚本要求环境中已安装 `ncu` 或
`nsys`；profiler 启动成本不计入程序内部 CUDA-event 时间。

公共异步 CUDA API 位于 `include/low_precision.cuh`。格式、公式、文件布局、
实测结果和优化分析见 [中文总结报告](docs/report.zh-CN.md)。
