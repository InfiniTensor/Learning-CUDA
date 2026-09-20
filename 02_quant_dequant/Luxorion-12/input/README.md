# tensor.bin 输入协议

`tensor.bin`由固定20字节header和紧随其后的行主序data组成。文件中不写入`[header]`、`[data]`或字段名称。

| 字节偏移 | 长度 | 字段 | 编码 |
|---:|---:|---|---|
| 0 | 8 | `num_rows` | little-endian `int64`，必须大于0 |
| 8 | 8 | `num_cols` | little-endian `int64`，必须大于0 |
| 16 | 4 | `dtype` | ASCII `fp32`或`fp16`，无结尾零字节 |
| 20 | 其余 | `values` | `num_rows * num_cols`个行主序元素 |

FP32 data使用little-endian IEEE-754 binary32；FP16 data使用little-endian IEEE-754 binary16。文件总大小必须严格等于：

```text
20 + num_rows * num_cols * (dtype == fp32 ? 4 : 2)
```

读取器拒绝非正尺寸、乘法或文件长度溢出、未知dtype、截断/多余data以及NaN/Inf。FP16在Host读取后转换为FP32，再交给CPU reference或上传GPU；矩阵只按行主序展平，不改变元素顺序。

## 量化分块约定

当前baseline采用连续展平方案：先把行主序矩阵视为长度`num_rows * num_cols`的一维数组，再按连续下标分块。MXFP8使用`block_id = i / 32`，NVFP4使用`block_id = i / 16`；到达行末时不强制结束量化块，因此块可以跨越相邻两行。`num_rows`和`num_cols`用于保存与恢复形状，不改变baseline的分块边界。

实现见`cpu_reference/include/io.h`与`cpu_reference/src/io.cpp`，协议和错误输入测试见`cpu_reference/tests/test_io.cpp`。

## 量化参数文件

`quant.cfg`严格支持：`format`、`block_size`、`scale_mode`、`output_type`、`rounding`、`target_gpu`，以及可选的`seed`。`seed`只用于stochastic rounding，默认1234。未知、重复、缺失字段和格式不匹配的block size会被拒绝。
