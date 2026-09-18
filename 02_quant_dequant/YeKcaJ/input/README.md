# input：输入张量

存放喂给量化程序的输入矩阵。**只放数据，不放源码。**

---

## 目录结构

```text
input/
├── <月日>/<编号>.fp32|.fp16        普通功能输入，由 generate 生成
├── <月日>/<编号>.<dtype>.json      该输入的形状、分布、种子与 sha256
├── benchmark-v1/                   固定性能输入（正式优化必须复用）
└── evaluation-v1/                  固定误差评估输入
```

---

## 输入文件格式

题目第3页给出行数、列数、dtype和行主序数据的逻辑定义。本程序将其序列化为小端二进制：

```text
[header]
magic: char[8]          FP32INP1 或 FP16INP1，同时标识 dtype
version: uint32         1
num_rows: uint64
num_cols: uint64

[data]
values: dtype[num_rows * num_cols]
```

头部共28字节，不能按题目示意写入一个可变长dtype字符串。**不能把无头原始数组改扩展名后使用**。程序会校验头部、形状与长度，并拒绝 NaN/Inf。具体布局见 `Core/io/tensor_io.h::read_input`。

---

## 生成普通输入

在项目根目录执行：

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 1024 --cols 1024
```

产物：

```
input/<月日>/<编号>.fp32
input/<月日>/<编号>.fp32.json
```

终端会打印实际路径与编号。

| 参数 | 可选值 | 默认 |
|---|---|---|
| `--dtype` | `fp32`、`fp16` | `fp32` |
| `--distribution` | `uniform`、`normal`、`outlier` | `normal` |
| `--rows`、`--cols` | 任意正整数 | — |
| `--seed` | 整数 | 1234 |
| `--output` | 手动指定路径 | 按日期/编号自动分配 |

FP16 不使用强制的 2 的幂尺寸，任意行列数均可，例如 `--rows 128 --cols 129`。

---

## benchmark-v1：固定性能输入

```
benchmark-v1/normal_1048576.fp32     1M 元素
benchmark-v1/normal_4194304.fp32     4M 元素
benchmark-v1/normal_16777216.fp32    16M 元素
benchmark-v1/manifest.json           三个文件的 sha256
```

正式优化轮次必须使用这里的输入，`benchmark.py` 会校验 sha256，不一致会报错并停止。

**注意：这些 `.fp32` 文件被 `.gitignore` 排除，不在版本库里**（三个文件合计约 84 MB）。而 `manifest.json` 入库，记录了正确哈希。

因此在新克隆的仓库里跑正式 benchmark 会提示固定输入缺失，这是预期行为。需要用 `--benchmark-export` 重建后再跑：

```bash
./Core/build/pipeline_mxfp8 --benchmark-export 1048576 /tmp/n1.fp32
./Core/build/pipeline_mxfp8 --benchmark-export 4194304 /tmp/n4.fp32
./Core/build/pipeline_mxfp8 --benchmark-export 16777216 /tmp/n16.fp32
```

把生成的三个文件按上表命名放回 `benchmark-v1/`，再用 `manifest.json` 里的 sha256 校验是否一致。完整说明见 [固定协议](../docs/BENCHMARK_PROTOCOL.md)。

---

## evaluation-v1：固定误差评估输入

误差评估实际读取这里的六份文件；CUDA/MUSA共享同一套输入。形状均为128×129，生成seed=1234，每种分布提供FP32和FP16两种类型：

```
evaluation-v1/
├── uniform.fp32 / uniform.fp16   均匀随机 U(-3,3)
├── normal.fp32 / normal.fp16     标准正态 N(0,1)
├── outlier.fp32 / outlier.fp16   正态数据，首元素+1000、末元素-1000
└── manifest.json                 形状、种子、分布和每个文件的SHA256
```

只准备输入（不需要GPU）：

```bash
python3 Core/tools/quantize.py prepare-evaluation
```

`evaluate` 也会自动准备输入：已有文件先校验再复用，缺失文件按同一规则补齐；已有文件或清单不匹配则报错，不覆盖。跨平台复制整个目录并校验SHA256；若标准库生成结果不同，需排查运行环境，不能篡改清单。

```bash
python3 Core/tools/quantize.py evaluate --backend cuda
```

结果写入 `output/evaluation-v1/cuda/`，重复运行自动使用 `cuda-2/`、`cuda-3/`，MUSA独立编号。输入不会写到output或records。六份二进制被Git忽略，`manifest.json`保留；新克隆后可执行上述准备命令重建。

---

## 相关文档

- 输出文件的三类交付形式见 [output/README.md](../output/README.md)
- 性能测试的固定条件见 [docs/BENCHMARK_PROTOCOL.md](../docs/BENCHMARK_PROTOCOL.md)
