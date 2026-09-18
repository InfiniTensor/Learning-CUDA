# 二进制文件与日志格式

本文固定本项目输入张量、低精度权重、反量化张量和运行日志的物理格式。它与 [format_spec.md](format_spec.md) 分工如下：前者定义 MXFP8/NVFP4 的数值语义，本文定义这些数值如何写入文件。

第一版格式的目标是：文件可以独立读取、字段不依赖 C/C++ struct 的内存对齐、NVFP4 真正按两个元素每字节存储，并可在读取时检测常见的损坏或配置不匹配。

## 通用规则

- 文件格式版本为 `1`。所有多字节整数与 FP16/BF16/FP32 bit pattern 均为 **little-endian**。
- 写入器必须逐字段序列化，不能直接对含 padding 的 C/C++ struct 使用 `fwrite`。读取器也必须按本文 offset 逐字段解析。
- 本项目只写入 IEEE 754 二进制浮点 bit pattern。第一版不接受大端文件；header 的 `byte_order` 不是 `1` 时，读取器必须报错。
- `num_rows`、`num_cols` 和 `element_count` 均为无符号 64 位整数，且必须满足 `element_count = num_rows * num_cols`，乘法不得溢出。
- 张量 payload 均按 row-major 连续排列，不写入每行 padding。线性下标为 `n = row * num_cols + col`。
- 所有保留字段、对齐 padding 都必须写零；第一版读取器忽略其数值，但必须检查已知 section 不重叠且位于文件范围内。

## 枚举编码

header 只存紧凑的枚举值；对应的配置文本字符串由程序在读写边界转换。

| 字段 | 数值 | 含义 |
| --- | ---: | --- |
| `byte_order` | `1` | little-endian。 |
| `dtype` | `1` | FP16。 |
| `dtype` | `2` | BF16。 |
| `dtype` | `3` | FP32。 |
| `tensor_role` | `0` | 输入张量。 |
| `tensor_role` | `1` | 反量化输出张量。 |
| `format` | `1` | MXFP8-E4M3 + E8M0。 |
| `format` | `2` | NVFP4-E2M1 + E4M3 + FP32 global scale。 |
| `scale_mode` | `1` | `tensor`；仅 MXFP8 合法。 |
| `scale_mode` | `2` | `block`。 |
| `rounding` | `1` | `nearest`（RNE）。 |
| `rounding` | `2` | `stochastic`。 |
| `local_scale_type` | `1` | E8M0，一个字节一个 scale。 |
| `local_scale_type` | `2` | E4M3，一个字节一个 scale。 |
| `scale_layout` | `1` | rowwise block 顺序；tensor 模式也使用此值。 |

## 普通张量文件：`QDTENSOR`

输入张量和反量化张量使用同一种容器。文件开头的 **64 字节固定 header** 的 `0--7` 字节必须为 ASCII `QDTENSOR`；header 之后是连续的元素数据。

### header（64 字节）

| Offset | 大小 | 字段 | 规则 |
| ---: | ---: | --- | --- |
| `0` | 8 | `magic` | ASCII `QDTENSOR`。 |
| `8` | 2 | `version` | `uint16`，值为 `1`。 |
| `10` | 2 | `header_bytes` | `uint16`，值为 `64`。 |
| `12` | 1 | `byte_order` | 值为 `1`。 |
| `13` | 1 | `dtype` | 由 `tensor_role` 决定，见下方约束表。 |
| `14` | 1 | `tensor_role` | 输入为 `0`，反量化输出为 `1`。 |
| `15` | 1 | `reserved0` | 写零。 |
| `16` | 8 | `num_rows` | `uint64`。 |
| `24` | 8 | `num_cols` | `uint64`。 |
| `32` | 8 | `element_count` | `uint64`，必须等于行数乘列数。 |
| `40` | 8 | `data_offset` | `uint64`；版本 1 固定为 `64`。 |
| `48` | 8 | `data_bytes` | `uint64`，必须等于元素数乘 dtype 字节数。 |
| `56` | 8 | `reserved1` | 写零。 |

数据区从 `data_offset` 开始，存储 `element_count` 个连续元素：FP16 与 BF16 每元素 2 字节，FP32 每元素 4 字节。反量化输出文件必须把 `tensor_role` 写为 `1`，但其 payload 仍是纯 row-major 数值数组；解释该文件不需要知道原量化格式。

### `tensor_role` 与 `dtype` 约束

虽然 `dtype` 枚举定义了 FP16、BF16 和 FP32 三种编码，**具体允许哪一种由
`tensor_role` 决定**。这是 QDTENSOR v1 的方向性约束，不是配置文件中的可选项。

| `tensor_role` | 合法 `dtype` | 用途 |
| --- | --- | --- |
| `input = 0` | FP16、FP32 | 待量化的原始矩阵。读取后会扩展为 host FP32，供 scale 计算和元素编码使用。BF16 输入不属于第一版需求，读取器必须拒绝。 |
| `dequantized_output = 1` | FP16、BF16、FP32 | 已反量化的输出矩阵。反量化计算先得到 host FP32，写入器再按目标 `dtype` 转换为对应物理 payload。 |

QDWGT header 中的 `source_dtype` 记录其来源输入文件的物理类型，因此同样**只允许
FP16 或 FP32**；它不记录 `output_type`。`output_type` 属于反量化操作，可让同一份
QDWGT 分别写出 FP16、BF16 或 FP32 的 QDTENSOR 输出。

### 例子

一个形状为 `2 × 3` 的 FP32 输入张量：`data_offset = 64`、`data_bytes = 24`，文件总大小为 `88` 字节。payload 顺序为 `x[0,0]`、`x[0,1]`、`x[0,2]`、`x[1,0]`、`x[1,1]`、`x[1,2]`。

## 低精度权重文件：`QDWGT`

量化输出使用独立容器。其 **128 字节固定 header** 的 `0--7` 字节为 ASCII `QDWGT`，后面补 3 个 `0x00` 到 8 字节；header 之后依次是 packed payload、可选零 padding 与局部 scale 数组。

文件的 section 排列是：

```text
0                         128                 payload_end       local_scale_offset
+-------------------------+-------------------+-----------------+------------------+
| 固定 header（128 字节）  | packed payload    | 零填充（0--7 B） | local scale 数组 |
+-------------------------+-------------------+-----------------+------------------+
```

`payload_offset`、`payload_bytes` 与 `local_scale_offset` 始终以 header 中记录的数值为准。版本 1 的写入器固定 `payload_offset = 128`，并将 local scale section 向上对齐到 8 字节；使用 offset 而非隐式推导，是为了让后续版本可以扩展 header 或增加 section。

### header（128 字节）

| Offset | 大小 | 字段 | 规则 |
| ---: | ---: | --- | --- |
| `0` | 8 | `magic` | ASCII `QDWGT`，后随 3 个零字节。 |
| `8` | 2 | `version` | `uint16`，值为 `1`。 |
| `10` | 2 | `header_bytes` | `uint16`，值为 `128`。 |
| `12` | 1 | `byte_order` | 值为 `1`。 |
| `13` | 1 | `format` | `1` 为 MXFP8，`2` 为 NVFP4。 |
| `14` | 1 | `scale_mode` | `1` 为 tensor（仅 MXFP8 合法），`2` 为 block。 |
| `15` | 1 | `rounding` | `1` 为 nearest，`2` 为 stochastic。 |
| `16` | 1 | `source_dtype` | 原输入的 FP16 或 FP32。 |
| `17` | 1 | `payload_element_bits` | MXFP8 为 `8`，NVFP4 为 `4`。 |
| `18` | 1 | `local_scale_type` | MXFP8 为 E8M0，NVFP4 为 E4M3。 |
| `19` | 1 | `scale_layout` | 版本 1 固定为 `1`，rowwise。 |
| `20` | 4 | `block_size` | `uint32`；MXFP8 为 32，NVFP4 为 16。 |
| `24` | 8 | `num_rows` | `uint64`。 |
| `32` | 8 | `num_cols` | `uint64`。 |
| `40` | 8 | `element_count` | `uint64`，必须等于行数乘列数。 |
| `48` | 8 | `payload_offset` | `uint64`；版本 1 写入器为 `128`。 |
| `56` | 8 | `payload_bytes` | `uint64`，见下文。 |
| `64` | 8 | `local_scale_offset` | `uint64`，tensor/block 均存在局部 scale section。 |
| `72` | 8 | `local_scale_count` | `uint64`，见下文。 |
| `80` | 8 | `local_scale_bytes` | `uint64`；版本 1 必须等于 `local_scale_count`。 |
| `88` | 4 | `global_scale` | IEEE 754 `float32` decode scale；仅 NVFP4 使用。MXFP8 必须写 `1.0f`。 |
| `92` | 4 | `flags` | bit 0 表示最后 payload 字节的高 nibble 已清零；其余 bit 为 0。MXFP8 的 `flags` 必须为 0。 |
| `96` | 8 | `stochastic_seed` | `uint64`；`rounding = nearest` 时写 0。 |
| `104` | 24 | `reserved` | 全部写零。 |

### payload 与 scale section

MXFP8 的 `payload_bytes` 等于 `element_count`；第 `n` 个字节就是线性元素 `n` 的 E4M3 编码。

NVFP4 的 `payload_bytes` 等于 `ceil(element_count / 2)`。线性偶数下标在低 nibble，奇数下标在高 nibble：

$$
\operatorname{byte}[n/2] =
\begin{cases}
(\operatorname{byte}[n/2]\ \&\ 0xf0)\ |\ q_n, & n\text{ 为偶数} \\
(\operatorname{byte}[n/2]\ \&\ 0x0f)\ |\ (q_n \ll 4), & n\text{ 为奇数}.
\end{cases}
$$

元素总数为奇数时，最后一个字节的高 nibble 必须为零，且 `flags & 1` 必须为 `1`。元素总数为偶数时，`flags & 1` 必须为 `0`。这使读取器可以检测尾 nibble 未初始化的问题。

局部 scale section 直接存储 one-byte scale code，顺序与 rowwise block 顺序一致。第 `b` 个 scale 对应的 block 为：

```text
row = b / ceil(num_cols / block_size)
block_in_row = b % ceil(num_cols / block_size)
col_begin = block_in_row * block_size
```

scale 个数规则如下：

| 格式与模式 | `local_scale_count` | scale 内容 |
| --- | ---: | --- |
| MXFP8 + `block` | `num_rows * ceil(num_cols / 32)` | 每个 block 一个 E8M0 字节。 |
| MXFP8 + `tensor` | `1` | 整张张量一个 E8M0 字节。 |
| NVFP4 + `block` | `num_rows * ceil(num_cols / 16)` | 每个 block 一个 E4M3 字节。 |

NVFP4 的 `global_scale` 是 `format_spec.md` 中定义的 FP32 **解码方向** scale；MXFP8 没有这一层 scale，因此固定写入 `1.0f`。`format = NVFP4` 与 `scale_mode = tensor` 是非法 header 组合，读取器必须在使用任何 section 前拒绝它。其余量化语义，包括 E8M0 的 NaN 和 E4M3 的饱和，均以 [format_spec.md](format_spec.md) 为准。

### 写入与读取校验

写入量化权重时必须校验：

- `format`、`payload_element_bits`、`local_scale_type` 和 `block_size` 的组合符合格式规范；
- `format = NVFP4` 时 `scale_mode` 必须为 `block`；
- `payload_bytes` 与 `element_count` 的关系正确；
- `local_scale_bytes = local_scale_count`，且 scale count 与 `scale_mode` 一致；
- `payload_offset >= header_bytes`，各 section 不重叠，`local_scale_offset` 满足 8 字节对齐；
- NVFP4 的 `global_scale` 为有限正数；MXFP8 的值为 `1.0f`；
- NVFP4 尾 nibble 与 `flags` 一致。

读取器不得根据本机 struct 的 `sizeof`、当前配置文件或文件名推断格式；必须只相信通过校验的 header。若版本号、magic、字段组合或文件长度不正确，应返回带原因的错误，不应启动 CUDA kernel。

## 误差与性能日志

日志使用 UTF-8 JSON 文件，每次命令执行覆盖指定的 `--report` 路径并写入一个 JSON object。JSON 比 CSV 更适合同时记录嵌套配置、文件大小和多项性能指标；后续实验汇总脚本可读取多个 JSON 文件。

必填结构如下。数值必须以 JSON number 写入，不能把浮点数转成字符串。

```json
{
  "schema_version": 1,
  "input": {
    "rows": 1024,
    "cols": 1024,
    "dtype": "fp32"
  },
  "config": {
    "format": "mxfp8",
    "block_size": 32,
    "scale_mode": "block",
    "rounding": "nearest",
    "stochastic_seed": 0,
    "output_type": "fp16",
    "target_gpu": "RTX 4060"
  },
  "artifacts": {
    "quantized_file_bytes": 1081472,
    "dequantized_file_bytes": 2097216,
    "payload_bytes": 1048576,
    "local_scale_bytes": 32768,
    "global_scale_bytes": 0
  },
  "error": {
    "max_abs": 0.0,
    "mae": 0.0,
    "mse": 0.0
  },
  "compression": {
    "input_payload_bytes": 4194304,
    "logical_quantized_bytes": 1081344,
    "logical_compression_ratio": 3.8788,
    "on_disk_compression_ratio": 3.8783
  },
  "performance": {
    "quant_kernel_ms": null,
    "dequant_kernel_ms": null,
    "quant_effective_bandwidth_gbps": null,
    "dequant_effective_bandwidth_gbps": null
  }
}
```

`max_abs`、`mae` 与 `mse` 均以原始输入转换得到的 FP32 值为参考。`logical_quantized_bytes` 只计低精度 payload、局部 scale 和 NVFP4 的 4 字节 global scale，不包括 header 和对齐 padding；`on_disk_compression_ratio` 则用完整量化文件大小计算。因此：

$$
\operatorname{logical\_compression\_ratio}
= \frac{\operatorname{input\_payload\_bytes}}
       {\operatorname{logical\_quantized\_bytes}}.
$$

有效带宽使用十进制 GB/s，并且只统计 CUDA kernel 的逻辑读写量，不包含文件 I/O、CPU/GPU 拷贝、内存分配、warmup 或 CUDA 初始化：

- 量化的逻辑字节数 = 输入 payload + 量化 payload + 局部 scale + NVFP4 global scale。
- 反量化的逻辑字节数 = 量化 payload + 局部 scale + NVFP4 global scale + 反量化输出 payload。
- 带宽 = 逻辑字节数 / kernel 时间；当时间以毫秒记录时，`GB/s = bytes / (ms * 1e6)`。

若量化与反量化未实际执行 CUDA kernel（例如前期 CPU reference 测试），对应时间与带宽字段写 `null`，不能伪造为 `0.0`。若 CUDA Event 因有限分辨率将一个极短 kernel 测为 `0.0 ms`，时间字段保留 `0.0`，但对应带宽字段写 `null`，避免除以零伪造无限带宽。

## 格式演进

- 增加字段、压缩方式或新的 format profile 时，必须提升 `version`，并保留旧版本读取逻辑或明确拒绝旧版本。
- 不能在不升版本的情况下改变枚举值、header offset、payload nibble 顺序、scale 顺序或 `global_scale` 的解码方向。
- 写入器只产生本文定义的版本 1；读取器可以先只支持版本 1，并在遇到其他版本时明确报错。
