# MXFP8 / NVFP4 软件量化与反量化

题目 2，训练营 ID：曹泽阳；提交目录：`zmuxuny`。默认程序为纯 CUDA 软件实现，编译目标为 Turing `sm_75`，不使用硬件 FP8/FP4 转换指令或 Tensor Core。数值编码、缩放、打包、解包均由本项目实现。本目录可独立构建、测试和提交。

[总结报告](REPORT.md) 汇总功能、正确性、性能及平台结论；[工具分析](PROFILING.md) 给出 Profiler / Sanitizer 证据；[提交要求对应表](SUBMISSION.md) 列出交付内容。

## 构建与运行

依赖 CUDA Toolkit >= 11.0、兼容的 C++17 host 编译器、Make；测试需要 Python 3 和 NumPy。CUDA 11.5 与本机 GCC 11 的标准库存在编译兼容问题，Makefile 优先选择已安装的 `g++-10`；其他机器可用 `HOSTCXX=/path/to/g++` 指定。

```bash
cd 02_quant_dequant/zmuxuny
make
mkdir -p results/tmp
python3 tests/generate.py --rows 1024 --cols 1024 --dtype fp32 \
  --distribution outliers --output results/tmp/input.bin
./build/quantize --input results/tmp/input.bin --config configs/nvfp4.toml \
  --packed results/tmp/weights.bin --output results/tmp/restored.bin \
  --log results/tmp/metrics.json --repeats 100
# 仅从保存的权重文件反量化，不需要原始输入或配置文件
./build/quantize --mode dequantize --packed results/tmp/weights.bin \
  --output results/tmp/reloaded.bin --output_type bf16 --log results/tmp/reload.json
python3 tests/validate.py
python3 tests/benchmark.py
```

`make test` 运行较短的参数组合测试；完整 `validate.py` 还覆盖所有 FP8 有限码值、FP4 码值、舍入中点及其两侧、零值、大动态范围、奇数列、尾块和文件错误。测试与正式程序分目录存放。

切换目标架构时先 `make clean`，再例如 `make ARCH=86`。默认 `sm_75` 二进制及 PTX 已在 RTX 3060 上运行；这证明编译路径不依赖 Ampere 指令，T4 的实际运行性能仍需在 T4 上测量。

## 平台构建与实验

| 平台 | 构建 | 二进制目录 | 实测与分析 |
|---|---|---|---|
| NVIDIA RTX 3060 / 4090 D | `make ARCH=86` / `make ARCH=89` | `build/` | [4090 D 报告](REPORT_4090D.md) |
| MetaX C500 | `make PLATFORM=metax` | `build/metax/` | [C500 报告](REPORT_C500.md) |
| Iluvatar 智铠 100（MR-V100） | `make PLATFORM=iluvatar` | `build/iluvatar/` | [MR-V100 报告](REPORT_ILUVATAR.md) |
| Moore Threads MTT S4000 | `make PLATFORM=musa` | `build/musa/` | [S4000 报告](REPORT_MUSA.md) |
| Ascend 910B2 | `make PLATFORM=ascend` | `build/ascend/` | [昇腾报告](REPORT_ASCEND.md) |

天数实测 CoreX 4.4.0，默认 `COREX_PATH=/usr/local/corex`、`IVCORE_ARCH=ivcore11`；使用 CoreX clang 编译，运行前设置 `LD_LIBRARY_PATH=$COREX_PATH/lib64:${LD_LIBRARY_PATH:-}`。`make PLATFORM=iluvatar test` 包含数值验证和 49,152 项随机哈希一致性检查。复现性能、Profiler 和 Sanitizer 的命令见对应平台报告。

摩尔线程实测 MUSA 4.3.6 / `mp_22`，默认 `MUSA_PATH=/usr/local/musa`；运行前设置 `MUSA_VISIBLE_DEVICES=0` 和 `LD_LIBRARY_PATH=$MUSA_PATH/lib:${LD_LIBRARY_PATH:-}`。完整数值验证、参数扫描及工具采样命令见 [S4000 报告](REPORT_MUSA.md)。

昇腾实测 ARM64 / CANN 9.0 / 910B2，使用 Ascend C 独立内核和 ACL 运行时。先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，再运行 `make PLATFORM=ascend test`；测试、性能对照及 msprof / mssanitizer 说明见 [昇腾报告](REPORT_ASCEND.md)。

## 配置语义

| 参数 | 支持值 / 含义 |
|---|---|
| `format` | `mxfp8` / `nvfp4` |
| `block_size` | 默认分别为 32 / 16；支持 16 的倍数，范围 16–1024 |
| `scale_mode` | `block` 为每行独立分块；`tensor` 为全张量共享一个局部缩放码 |
| `output_type` | `fp16` / `bf16` / `fp32` |
| `rounding` | `nearest` 为 ties-to-even；`stochastic` 按相邻可表示值距离随机选择 |
| `seed` | 随机舍入种子，默认 42；与元素线性索引共同决定结果 |
| `target_gpu` | 仅记录目标机器说明；日志的 `gpu` 是实际运行设备 |

默认块尺寸对应标准格式；其他块尺寸与 per-tensor 模式是本题要求的缩放对照扩展，不能当作硬件原生 MXFP8/NVFP4 块布局。输入限定有限 FP32/FP16 数值；全零块、尾块、负零均有定义。输出类型应能容纳反量化数值，例如大幅值 FP32 输入应选择 FP32 输出。

## 数值与布局

MXFP8 采用 E4M3 有限数编码，最大幅值 448；每 32 个连续元素默认保存 1 字节 E8M0 缩放码。设块最大幅值为 `a`，选择 `e = clamp(ceil(log2(a/448)), -127, 127)`，保存 `e+127`，反量化为 `decode_E4M3(q) * 2^e`。全零块使用缩放 1。向上取整策略参照 [NVIDIA MXFP8 说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)。

NVFP4 的 E2M1 正幅值码本为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，符号位在 bit 3。全局缩放 `g = max(abs(x))/(448*6)` 存为 FP32；局部缩放 `s = E4M3_RNE((block_amax/g)/6)` 存为 1 字节。反量化为 `(decode_E2M1(q) * decode_E4M3(s)) * g`，与 [NVIDIA NVFP4 定义](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) 一致。全零张量设 `g=1`；极小全局缩放下限为 FP32 最小正规数，避免全局除法溢出。局部缩放舍入为零时该块量化为零。

元素舍入使用配置指定策略；缩放因子始终确定性计算。溢出到格式范围外时饱和到最大有限值；低于最小非零值时按指定策略舍入。FP8 不产生 NaN 编码。

NVFP4 每个字节保存同一行的两个元素：偶数列在低 4 位，奇数列在高 4 位。奇数列数的行最后一个高半字节填零，因此 packed data 大小为 `rows * ceil(cols/2)`。缩放块不会跨行，最后一个不足块长的块仅统计有效元素。

### 二进制文件协议 v1

所有整数/浮点字段为 little-endian，无隐式结构体 padding。题目仅给出逻辑字段，未指定字符串长度和物理头格式，本实现明确以下协议，并提供 Python 读写工具。

张量头长 32 字节，Python struct 格式 `<8sQQII`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPTENS1\0` |
| 8 | uint64 | rows |
| 16 | uint64 | cols |
| 24 | uint32 | dtype：0=FP32，1=FP16，2=BF16 |
| 28 | uint32 | reserved=0 |
| 32 | dtype[] | 行主序数据，无 padding |

packed 头长 56 字节，Python struct 格式 `<8sQQIIIfQQ`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPPACK1\0` |
| 8 / 16 | uint64 / uint64 | rows / cols |
| 24 | uint32 | format：0=MXFP8，1=NVFP4 |
| 28 | uint32 | block_size |
| 32 | uint32 | tensor scaling：0/1 |
| 36 | float32 | global scale；MXFP8 固定 1 |
| 40 / 48 | uint64 / uint64 | scale 字节数 / packed data 字节数 |
| 56 | uint8[] | scale 数组，随后是 packed data |

BF16 以 16 位原始位模式存储，使用整数实现 RNE 转换，在 `sm_75` 上可运行。文件格式为线性软件交换格式，不是 Blackwell GEMM 的 swizzled 硬件布局。

## 验证与计时

主程序运行 scalar C++ CPU 参考并逐字节比较 GPU 输出。独立 Python oracle 用显式码本搜索实现编码，避免与 CUDA 编码算法同源。

`quant_ms` 包含所需的全局归约、局部 scale kernel 和打包 kernel；`dequant_ms` 为解包反量化 kernel。每组预热 3 次，用 CUDA event 统计多次执行的均值。`gpu_with_transfers_ms` 包含 H2D 输入、量化、反量化、D2H 输出，不含文件 I/O、内存分配和正确性检查；对应 CPU 时间是单线程 scalar 量化加反量化。基准脚本每项启动三个独立进程并报告中位数，同时保留全部原始日志。显存数据热缓存，未锁 GPU 时钟。

`quant_effective_gbps` 与 `dequant_effective_gbps` 按逻辑输入、输出和 scale 字节数计算，不代表 profiler 测得的 DRAM 流量。压缩率分别报告 payload（含全部 scales 和 global FP32）以及完整文件（含 header）；不能仅用 32/4 推算实际压缩率。

实测结果及分析见 [REPORT.md](REPORT.md)，原始数据见 [results/](results/)。

工具检查脚本为 `tests/profile.py`，可用 `--ncu`、`--sanitizer` 指定可用工具路径；仅检查本题。本机 WSL 的 profiler / 调试接口限制及失败日志单独记录，不计作检查通过。`python3 tests/report.py` 从已提交 JSON 重建总收益数据及平台性能表；总加速比使用起始基线与最终实现的耗时直接计算。

原生 Linux 分析结果及本机 WSL 设置说明见 [PROFILING.md](PROFILING.md)。

## 优化实现与原生 Linux 分析

默认块长度的 scale 计算和量化写出已融合，整块对齐张量使用连续 tile，奇数列和行尾保持独立打包。E4M3 编码/解码使用整数位域，E8M0 使用精确二次幂缩放；NVFP4 最近偶数舍入使用缩放后的精确阈值，随机舍入保留原有商与概率计算。

```bash
make clean && make ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++
python3 tests/validate.py
python3 tests/profile_native.py
python3 tests/report_4090.py
```

大张量前后交替比较可用 `tests/benchmark_extended.py --before /path/to/old/quantize --after build/quantize`。`tests/report_4090.py` 生成 NVIDIA 最终性能报告；原始测量数据按设备保存。

## 软件优化与可选硬件转换

NVIDIA 后端量化在整块对齐输入上使用每线程 4 元素向量读写，反量化按输出类型使用 4/8 元素；全局 amax 使用向量加载和 CTA 归约。FP8 软件最近偶数舍入通过整数进位完成。BF16 在 Ampere 及更新架构使用原生转换，较旧架构保留位运算实现。详情、消融与全部实测见 [4090 D 实验报告](REPORT_4090D.md)。

额外的 Ada FP8 转换对照使用独立可执行文件，不改变默认程序；需要 CUDA >=12.1 和 `sm_89` 或更新：

```bash
make native ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++
python3 tests/validate.py --binary build/quantize_native
```

原生对照只加速 E4M3 最近偶数转换；随机舍入和 NVFP4 的 E2M1 编码继续使用软件实现。日志 `fp8_encoding` 标识所用实现。两路的 packed 文件、scales 和舍入语义一致。
