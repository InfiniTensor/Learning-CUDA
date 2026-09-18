# CUDA 输出格式与验证

> 2026-09-10：当前核心迁到 [Core](../Core/README.md)。本目录的 pipeline/output 和 CPU 桥接文件仅转发；旧 benchmark/regression 保留。下面是历史记录，不是当前源码位置索引。

2026-09-09 更新：新增可配置算子 `pipeline.cuh`、文件协议 `pipeline_io.h` 和入口
`pipeline_main.cu`。本目录现在不只包含输出辅助代码，也包含 v2 扩展算子。
详见 [项目 README](../README.md) 和 [最终报告](../FINAL_REPORT.md)。
两个工程现在各 29 个 CTest：保留 27 个旧测试，增加 pipeline_regression 和 pipeline_io。
下文 27 项和旧 benchmark 的描述保留作历史说明；新性能脚本为 `tools/benchmark.py`。

`output.cuh` 为 MXFP8、NVFP4 共用的输出类型转换、文件写出和结果比较代码。
两个项目均保持原有命令形式，在反量化命令末尾新增：

```text
--output-dtype fp32|fp16|bf16
```

默认 FP32。该参数不适用于量化命令。输入 golden 仍为冻结的 FP32DEQ1 文件。

## 输出规则

反量化中间运算保持 FP32，并在最终写入时转换一次。FP16/BF16 使用 nearest-even
舍入，保留符号零，支持 subnormal，超出目标有限范围时转换为同号无穷，不做有限值饱和。
NaN 按类别验证，不要求 payload 相同。

FP32 校验有限值绝对差不超过 `1e-6`，并检查符号零、NaN 和无穷的类别/符号。
FP16/BF16 校验 CUDA 结果与舍入后的 CPU golden 位模式一致，不能用原 FP32 的
`1e-6` 阈值判断正常的 16 位舍入误差。

## 二进制布局

所有整数和浮点 payload 为小端，矩阵按行展开。头部固定 36 字节：

```text
magic[8]  = FP32DEQ1 / FP16DEQ1 / BF16DEQ1
version   = uint32(1)
rows      = uint64
cols      = uint64
count     = uint64(rows * cols)
payload   = count 个目标类型元素
```

FP32 每元素 4 字节；FP16 和 BF16 每元素 2 字节。格式版本仍为 1，不修改 CPU
reference 的输入、量化格式或冻结 golden。两个新增 magic 避免把 16 位文件当作 FP32 读取。

## 执行测试

在项目根目录执行：

```bash
bash CUDACommon/run_tests.sh
```

脚本构建两个工程并执行 CTest，检查所有 CPU 输入和 golden 的 SHA256。每个工程
27 个 CTest 项；日志和输出在各自的 `build/tests/results/`。

`tests/reference.cpp` 在单独编译单元中包含原 CPU 源码，使用原量化函数产生新增用例的
预期值。编码测试比较所有相邻有限正 E4M3 值和 E2M1 幅值中点及其相邻 FP32 值，
包含正负两侧。反量化测试覆盖有效编码、负零、三种输出、FP16/BF16 下溢/溢出/舍入边界。
16 位期望值由 double 二进制缩放与 nearest-even 舍入独立计算，不复用 CUDA 转换实现。

## GPU 检查

通过环境变量指定可运行的 Compute Sanitizer 绝对路径：

```bash
SANITIZER_BIN=/absolute/path/to/compute-sanitizer bash CUDACommon/run_tests.sh
```

这会额外执行 memcheck（含 32 字节 padding 和泄漏检查）、racecheck、synccheck。
检查器发现问题时退出码为 99，脚本停止，不能把未启动的检查记为 PASS。

本机原 `/usr/bin/compute-sanitizer` 2022.4 无法完成注入；本次使用 NVIDIA 官方
CUDA 12.9.79 独立工具包，不替换系统 CUDA 12.0 编译器或驱动。下载来源：

```text
https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-12.9.79-archive.tar.xz
SHA256=e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf
```

本次可执行路径（临时目录，系统清理后需重新解压）：

```text
/tmp/nvfp4-sanitizer.UQoisO/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/compute-sanitizer/compute-sanitizer
```

## 正式性能测试

两个 CUDA 工程都生成 `benchmark`。它对 1M、4M、16M 个 FP32 元素分别预热 3 次并重复 20 次，输出 `median_ms`、`p95_ms` 和 `effective_GBps`：

```bash
./CUDAMXFP8/build/benchmark 20
./CUDANVFP4/build/benchmark 20
```

当前计时是端到端 wall-clock，包含每次调用的设备分配、释放和同步；结果与完整表格见 `TEST_RESULTS.md`。NVFP4 量化包含 global-max、主机端 global-scale 计算及拷回，因此其量化吞吐低于 MXFP8。后续优化方向是设备端归约和复用分配，且必须保持冻结 golden 的逐字节结果。
