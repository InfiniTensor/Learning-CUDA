# 实验配置矩阵

本目录中的每个 `.toml` 都是可以直接传给 `quant_dequant --config` 的完整 app 配置。
它们不是模板，也不会由脚本临时生成；`scripts/run_e2e_suite.py` 默认扫描全部 `.toml`
并逐一运行。

| 格式 | scale mode | rounding | 输出 dtype | 文件数 |
| --- | --- | --- | --- | ---: |
| MXFP8 | `block`、`tensor` | `nearest`、`stochastic` | `fp16`、`bf16`、`fp32` | 12 |
| NVFP4 | 仅 `block` | `nearest`、`stochastic` | `fp16`、`bf16`、`fp32` | 6 |

命名固定为：

```text
mxfp8_<scale_mode>_<rounding>_<output_type>.toml
nvfp4_block_<rounding>_<output_type>.toml
```

`nearest` 配置省略 `stochastic_seed`，因此 loader 使用默认值 0；`stochastic` 配置都
显式使用 `stochastic_seed = 20260917`，使同一输入与配置可以复现完全相同的编码结果。

例如，只跑 MXFP8 的 tensor scale、随机舍入和 BF16 输出：

```shell
python3 scripts/run_e2e_suite.py \
  --config-names mxfp8_tensor_stochastic_bf16 \
  --output-dir outputs/mxfp8_tensor_stochastic_bf16
```

完整字段语义及单方向/完整 app loader 的区别见
[`../docs/configuration.md`](../docs/configuration.md)。
