# CUDA MXFP8

本目录包含 CUDA MXFP8 的量化和反量化实现。量化格式与
`../CPUMXFP8` 的 CPU reference 保持一致：E4M3FN 数据、每 32 个元素一个
block、E8M0 block scale，格式版本为 `1`。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

项目默认使用 `sm_75`，可用 `-DCMAKE_CUDA_ARCHITECTURES=86` 覆盖；本机在 RTX 3060 Laptop 上验证。

## 反量化

从本目录执行：

```bash
./build/cuda_mxfp8 \
  ../CPUMXFP8/tests/golden/outlier_tail.mxfp8 \
  ../CPUMXFP8/tests/golden/outlier_tail.dequant.fp32 \
  tests/results/outlier_tail.cuda.dequant.fp32
```

程序会在 GPU 上逐元素解码 E4M3，并乘以所属 32 元素 block 的 E8M0 scale，
然后与 CPU 反量化 golden 进行逐元素比较。

## 量化

量化命令读取 CPU reference 生成的 `FP32INP1` 文件，并输出 `MXFP8Q1` 文件：

```bash
./build/cuda_mxfp8 --quantize \
  ../CPUMXFP8/tests/data/outlier_tail.fp32 \
  tests/results/outlier_tail.cuda.mxfp8 \
  ../CPUMXFP8/tests/golden/outlier_tail.mxfp8
```

最后一个参数是可选的 CPU golden。提供该参数时，程序会严格比较 CUDA 和 CPU 的：

- E4M3 `data` 字节；
- E8M0 `scales` 字节；
- 矩阵尺寸和 payload 长度。

量化 kernel 使用一个 CUDA block 处理一个 32 元素 quantization block：先在 shared
memory 中归约 `max(abs(x))`，再计算 E8M0 scale，最后由每个线程编码一个 E4M3 字节。

## 检查和性能

程序会检查：

- MXFP8Q1 magic 和 `format_version = 1`；
- data/scales 的长度和 block size；
- 量化输出与 CPU golden 的逐字节差异；
- 反量化输出与 CPU golden 的最大绝对差异；
- kernel-only 时间和有效带宽。

反量化逐元素比较阈值为 `1e-6`。量化的正确性验收要求 data 和 scales 的差异均为零。

固定测试输出保存在 `tests/results/`，不覆盖 CPU 的 `tests/golden/`。

本页描述旧版 v1 CLI。新增 FP16 输入、配置与缩放模式、优化编码和设备驻留性能测试
位于 `pipeline` 入口，见 [项目 README](../README.md) 和 [最终报告](../FINAL_REPORT.md)。

## FP16 / BF16 输出

在反量化命令末尾追加 `--output-dtype fp16` 或 `--output-dtype bf16`。默认保持 FP32。
例如从本目录执行：

```bash
./build/cuda_mxfp8 \
  ../CPUMXFP8/tests/golden/outlier_tail.mxfp8 \
  ../CPUMXFP8/tests/golden/outlier_tail.dequant.fp32 \
  build/tests/results/outlier_tail.cuda.dequant.fp16 --output-dtype fp16
```

kernel 先按 reference 的 FP32 运算顺序计算，再以 nearest-even 转换为目标类型，
直接写入 16 位显存数组。文件头区分 `FP16DEQ1` 和 `BF16DEQ1`，payload 每元素 2 字节。
验证用 CPU FP32 golden 转换后的目标类型逐位比较；不是要求 FP16 与原 FP32 零误差。

## 回归测试

```bash
ctest --test-dir build --output-on-failure
```

包含 27 个 CTest 项：量化、完整文件字节比较、三种反量化命令行输出，以及边界回归。
边界回归覆盖 139 组生成输入、800 个编码舍入样本和独立的 16 位转换检查。
自动测试文件放在 `build/tests/results/`。

统一测试、内存检查和输出文件格式说明见 [CUDACommon/README.md](../CUDACommon/README.md)。
