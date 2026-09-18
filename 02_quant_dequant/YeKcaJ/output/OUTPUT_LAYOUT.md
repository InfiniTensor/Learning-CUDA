# 输入、输出与记录目录

## `output/`：最终交付输出

题目要求的输出默认放在这里。用户通过 `--output-dir` 可直接指定实际保存目录：

```text
<指定目录>/<输入名>_<格式>_<backend>_<输入dtype>_<输出dtype>.lpq
<指定目录>/<输入名>_<格式>_<backend>_<输入dtype>_<输出dtype>.<输出dtype>
<指定目录>/<输入名>_<格式>_<backend>_<输入dtype>_<输出dtype>.json
```

格式/平台在文件名中隔离，重复运行加 `_2`、`_3`。`--save all` 默认保存三类文件，也可多选 `weights tensor log`。终端会显示实际保存位置。

不指定 `--output-dir` 或 `--prefix` 时，按输入日期、编号、格式和平台自动组织：

```text
output/<月日>/<输入编号>/<格式>/<backend>/
├── <输入dtype>_<输出dtype>.lpq         低精度权重：header + packed data + scale
├── <输入dtype>_<输出dtype>.<输出dtype>  反量化张量：按行主序二进制
└── <输入dtype>_<输出dtype>.json         误差、压缩率、kernel时间、带宽
```

自动布局用目录区分 CUDA/MUSA，指定输出文件夹时用文件名区分；已有结果不覆盖。

## `input/`：输入文件

```text
input/<月日>/<编号>.fp32|.fp16       普通功能输入
input/benchmark-v1/                  固定性能输入，只放 normal_N.fp32 和 manifest.json
input/evaluation-v1/                 固定误差评估输入：uniform、normal、outlier
```

`python3 Core/tools/quantize.py prepare-evaluation` 生成六份固定误差输入及 `manifest.json`；`evaluate --backend cuda` 也会自动准备并读取它们，已有输入必须通过哈希校验。

误差评估结果放在 `output/evaluation-v1/<backend>/`，重复运行使用 `<backend>-2`、`<backend>-3`：

```text
uniform.json / normal.json / outlier.json    三类分布分别统计
summary.json / status.json                  全部统计、完成状态
<分布>/<格式>/<入>_<出>_<缩放>_<舍入>.lpq   低精度权重
<分布>/<格式>/<入>_<出>_<缩放>_<舍入>.<出>  反量化张量
<分布>/<格式>/<入>_<出>_<缩放>_<舍入>.json  单组日志
```

输入与输出同名 `evaluation-v1` 表示同一套评估数据的两个环节，不是重复存储。三类交付文件都保留在输出目录，详细字段及操作见 [README](README.md)。

## `records/`：过程证据

只保存 benchmark 原始 JSONL、环境和哈希、nsys/NCU/检查工具日志、平台适配过程、历史版本和重复实验归档。它用于审计和复现，不是用户运行后寻找三类交付文件的目录。

可以清理 `output/` 旧结果和可重建的 `Core/build/`；不要删除 `input/benchmark-v1/` 或 `records/` 原始证据。
