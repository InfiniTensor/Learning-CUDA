# MXFP8 / NVFP4 Reference

当前目录实现不依赖新硬件的 CPU reference：

- MXFP8：E4M3 数据，按每行连续 32 个元素使用一个 E8M0 scale。
- NVFP4：E2M1 数据，按每行连续 16 个元素使用一个 E4M3 block scale，并保存 FP32 global scale。
- NVFP4 每两个 4-bit 值打包到一个字节，偶数索引使用低 4 bit。
- `NearestEven` 和 `Stochastic` 均提供接口；默认使用 `NearestEven`。
- `Fp16` 和 `Bf16` 使用明确的 16-bit storage wrapper；FP16 输入可直接量化，MXFP8/NVFP4 解量化可输出 FP32、FP16 或 BF16。

目录按共享代码和后端实现分层：`include/low_precision` 保存公共接口，`common` 保存 CPU reference 与文件格式，`backends/nvidia` 保存 NVIDIA CUDA kernel，`apps` 保存 CLI/benchmark。天数智芯、壁仞和昇腾后端分别位于 `backends/iluvatar`、`backends/biren`、`backends/ascend`，均复用公共接口和文件格式。

天数智芯（Iluvatar）使用 CoreX Clang 的 CUDA 兼容模式构建同一 kernel。设备端 FP16 输入按 storage bits 软件解码，避免不同后端的半精度转换指令造成 payload 差异。天数环境示例：

```bash
export COREX_VERSION=4.4.0
source /usr/local/corex-${COREX_VERSION}/enable
export LD_LIBRARY_PATH=/usr/local/corex-${COREX_VERSION}/lib64:${LD_LIBRARY_PATH:-}
ixsmi
cmake -S . -B build-iluvatar -G "Unix Makefiles" \
  -DLP_BACKEND=iluvatar \
  -DCMAKE_CXX_COMPILER=/usr/local/corex-${COREX_VERSION}/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/corex-${COREX_VERSION}/bin/clang++
cmake --build build-iluvatar
```

The `LD_LIBRARY_PATH` export is required when running the generated binaries
outside the CoreX environment. The Iluvatar path compiles the shared `.cu`
kernel source with CoreX Clang's `ivcore` target; it does not duplicate the
CPU reference or file format implementation.

壁仞使用 BIRENSUPA 编译相同的 CUDA 风格 kernel；该路径使用 `.su` adapter
和轻量 runtime 映射，不复制量化实现。已在 Biren166M（SUPA 1.11，`br100`）
完成构建与正确性验证：

```bash
source /usr/local/birensupa/br_container_tools/brsw_set_env.sh
cmake -S . -B build-biren -G Ninja \
  -DLP_BACKEND=biren -DLP_BIREN_ARCH=br100
cmake --build build-biren
```

昇腾后端使用 CANN Ascend C 实现量化 kernel，并复用公共 benchmark 与 CLI。
已在 Ascend 910B1（CANN 9.0）完成构建与正确性验证：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cmake -S . -B build-ascend -G "Unix Makefiles" \
  -DLP_BACKEND=ascend -DLP_ASCEND_SOC=Ascend910B1
cmake --build build-ascend -j2
```

构建：

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

CLI 使用 little-endian 原始输入和量化文件：

```bash
./build/quant_cli quantize --input input.bin --output tensor.qnt \
  --rows 1024 --cols 2048 --input-type fp32 --format nvfp4 \
  --rounding nearest
./build/quant_cli dequantize --input tensor.qnt --output output.bin \
  --output-type bf16
```

`.qnt` 文件头保存 magic、版本、矩阵尺寸、输入类型、量化格式、rounding、payload 大小、block size、scale mode 和 NVFP4 global scale；当前写入版本为 v2，同时兼容读取 v1 文件。读取时会校验 scale metadata、payload 尺寸并拒绝截断或尾随数据。

运行 GPU 正确性与性能 benchmark（默认 `1024 x 2048`，每个 case 重复 5 次）：

```bash
./build/benchmark [rows] [cols] [repeats]
```

运行覆盖小、中、大矩阵的尺寸 sweep：

```bash
./build/benchmark --sweep [repeats]
```

输出为 CSV，包含随机、正态、离群值输入的误差、压缩率、读写字节数、API wall time、CUDA Event device time、p50/min/std、有效带宽和 payload 一致性。CPU 行的 device time 及对应带宽填 0；GPU 行同时保留 API 端到端时间和 device-only 时间。

CSV 还包含 `input_type` 和 `output_type` 列；CUDA benchmark 会额外测量 FP16 输入到 FP16/BF16 输出的路径。CLI 文件读写端到端计时可运行：

```bash
./apps/benchmark_cli.sh ./build/quant_cli 1024 2048
```

真正走 CUDA kernel 的文件工作流使用 `cuda_quant_cli`：

```bash
./build/cuda_quant_cli quantize --input input.bin --output tensor.qnt \
  --rows 1024 --cols 2048 --input-type fp16 --format nvfp4 --rounding stochastic --seed 17
./build/cuda_quant_cli dequantize --input tensor.qnt --output output.bin \
  --output-type bf16
```

该 CLI 将输入复制到 GPU，调用 CUDA 量化/反量化 API，再将 packed payload、scale 或输出张量写回文件。CUDA 量化支持 `nearest` 和带显式 seed 的 `stochastic`；同一 seed 可复现相同 payload。CPU reference CLI 仍保留用于逐字节对照。

MXFP8 的 block scale 编码以及 NVFP4 的 block `amax`、global `amax`、block scale 编码和 packed payload 均在 CUDA 上完成；NVFP4 host 端只往返一个 global scale。

FP16 输入以及 FP16/BF16 输出路径在 CUDA kernel 内直接完成格式转换，不再分配 FP32 临时输入或输出缓冲区；核心计算仍使用 FP32 中间值，保证与 CPU reference 的 payload 逐字节一致。NVFP4 的 global scale 选择也在设备端完成，公开同步 API 只在需要返回 `global_scale` 时复制一个标量到主机。
