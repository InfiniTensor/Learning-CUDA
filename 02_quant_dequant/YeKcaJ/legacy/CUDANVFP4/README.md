# CUDA NVFP4 Packed Kernel

本页描述旧版 v1 CLI。新增配置入口 `pipeline` 支持 FP16 输入、tensor/block、随机舍入、
设备端两级全局归约和显存复用，见 [项目 README](../README.md) 与 [最终报告](../FINAL_REPORT.md)。

本目录实现 NVFP4 的 CUDA 量化、4-bit 打包和反量化。格式与
`../CPUNVFP4` CPU reference 以及 `../REFERENCE_SPEC.md` 保持一致。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

当前 CMake 默认使用 `sm_75`，可由命令行覆盖；本机在 RTX 3060 Laptop 上验证。

## CUDA 量化

每个 CUDA block 负责 16 个 NVFP4 元素。程序先用一个 reduction kernel 求整个
输入的最大绝对值，计算：

```text
global_scale = max(abs(input)) / (6 * 448)
```

然后量化 kernel 在每个 16 元素 block 内归约 block maximum，将归一化 block scale
编码为 E4M3，并将输入量化为 E2M1。每两个 nibble 打包成一个字节：偶数元素在低
4 bit，奇数元素在高 4 bit。

```bash
./build/cuda_nvfp4 --quantize \
  ../CPUNVFP4/tests/data/outlier_tail.fp32 \
  tests/results/outlier_tail.cuda.nvfp4 \
  ../CPUNVFP4/tests/golden/outlier_tail.nvfp4
```

最后一个参数可选。提供 CPU golden 时，程序会逐位比较 `global_scale`，并逐字节比较
packed data 和 E4M3 block scales。

## CUDA 反量化

```bash
./build/cuda_nvfp4 --dequantize \
  tests/results/outlier_tail.cuda.nvfp4 \
  ../CPUNVFP4/tests/golden/outlier_tail.dequant.fp32 \
  tests/results/outlier_tail.cuda.dequant.fp32
```

反量化过程从 packed byte 拆出 nibble，解码 E2M1，再乘以：

```text
effective_scale = global_scale * decode_e4m3(block_scale)
```

FP32 输出与 CPU golden 的最大绝对误差阈值为 `1e-6`。

新增 `--output-dtype fp16|bf16`，不传时保持 FP32。例如：

```bash
./build/cuda_nvfp4 --dequantize \
  ../CPUNVFP4/tests/golden/outlier_tail.nvfp4 \
  ../CPUNVFP4/tests/golden/outlier_tail.dequant.fp32 \
  build/tests/results/outlier_tail.cuda.dequant.bf16 --output-dtype bf16
```

kernel 按 FP32 计算 `global_scale * block_scale` 和反量化乘法，再以 nearest-even
转换成 FP16/BF16 并直接写入 16 位数组。比较目标是 CPU FP32 golden 舍入后的位模式。
FP16/BF16 文件 payload 每元素 2 字节，使用独立的格式标识。

## 边界处理与回归

- packed 写入前检查 `even_index < count`，最后不足 16 元素的 block 不会越界写入。
- 奇数个元素的最后一个高 nibble 固定为零。
- `effective_scale <= 0` 时按 CPU reference 编码正零，不执行除零。
- scale 为正时保留输入负零的 E2M1 符号；零 scale 分支输出正零。
- 比较量化数据前先检查形状和数组长度，避免错误 golden 导致比较越界。

```bash
ctest --test-dir build --output-on-failure
```

包含 27 个 CTest 项：量化、完整文件字节比较、三种反量化命令行输出，以及边界回归。
边界回归直接调用冻结 CPU reference，覆盖长度 0～65、255/256/257/1023/1024/1025、
负零、舍入为零的 block scale、E2M1 舍入中点，共 141 组生成输入；另检查 800 个编码
舍入样本和独立的 FP16/BF16 转换边界。自动测试输出位于 `build/tests/results/`。

## 固定测试结果

`zeros`、`basic`、`outlier_tail`、`tail_block`、`random` 五组测试均已通过：

```text
global_scale_mismatches=0
packed_byte_mismatches=0
block_scale_byte_mismatches=0
max_abs_diff=0
element_mismatches=0
```

CUDA 临时输出保存在 `tests/results/`，不会覆盖 CPU 的 golden 文件。

当前量化的 `kernel_ms` 包含全局最大值 kernel、最大值回传、CPU 计算 scale 的间隔和
packed kernel；不能当作纯 kernel 时间或正式性能结果。反量化计时仅包围反量化 kernel。

统一测试、内存检查和输出文件格式说明见 [CUDACommon/README.md](../CUDACommon/README.md)。
