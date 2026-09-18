# output：最终交付输出

依据[题目 PDF](../docs/2026夏季训练营%20CUDA%20方向项目.pdf)第 4 页，三类输出全部保存在这里：低精度权重、反量化张量、误差与性能日志。另外，随机矩阵、正态矩阵、含异常值矩阵分别提供误差统计。

本目录只保存程序产出的结果，不放源码。

`run` 默认保存全部三类文件；用户可用 `--save` 选择其中一类或多类。题目交付与 `evaluate` 批量评估保留全部三类。

---

## 目录结构

```text
output/
├── OUTPUT_LAYOUT.md                        目录格式规范
├── <月日>/<编号>/<格式>/<backend>/         单次运行的输出
│   ├── <入>_<出>.lpq                        低精度权重
│   ├── <入>_<出>.<出>                       反量化张量
│   └── <入>_<出>.json                       误差与性能日志
└── evaluation-v1/<backend[-次数]>/         误差评估的最终结果
    ├── uniform.json
    ├── normal.json
    ├── outlier.json
    ├── summary.json                        全部配置的统计
    ├── status.json                         完成状态及输入哈希
    └── <分布>/<格式>/                      每组实际结果
        ├── <入>_<出>_<缩放>_<舍入>.lpq
        ├── <入>_<出>_<缩放>_<舍入>.<出>
        └── <入>_<出>_<缩放>_<舍入>.json
```

`<格式>` 为 `mxfp8` 或 `nvfp4`，`<backend>` 为 `cuda` 或 `musa`，`<入>_<出>` 形如 `fp32_fp16`。

CUDA 与 MUSA 即使输入相同也分目录保存，避免互相覆盖。同一路径不覆盖已有结果，重复运行会分配新前缀。

---

## 三类输出说明

### ① 低精度权重文件 `.lpq`

二进制，布局为 72 字节头部、packed data、scale 数组。头部含格式版本 2、形状、缩放/舍入设置、global scale 和数据长度：

- MXFP8：E4M3 元素，每元素 1 字节；block 模式每 32 元素一个 E8M0 scale。
- NVFP4：E2M1 元素，**每字节存两个元素**（偶数元素低 4 bit，奇数元素高 4 bit）；block 模式每 16 元素一个 E4M3 scale，整个张量另有一个 FP32 global scale。
- tensor 模式两种格式都使用一个覆盖全张量的分组。NVFP4 奇数长度最后一个字节的高 4 bit 补零。

### ② 反量化张量 `<出>`

二进制，36 字节头部（magic、version、rows、cols、count）后接行主序数据。数据类型由配置的 `output_type` 决定，可为 `fp32` / `fp16` / `bf16`，每元素分别占 4 / 2 / 2 字节；读取时不能把头部当成张量元素。

### ③ 误差与性能日志 `.json`

包含题目要求的全部指标：

| 字段 | 含义 |
|---|---|
| `max_abs_error` | 最大绝对误差 |
| `mae` | 平均绝对误差 |
| `mse` | 均方误差 |
| `compression_payload` | 有效数据压缩率 |
| `compression_file` | 完整文件压缩率 |
| `quant_kernel_ms` | 量化设备序列时间，包含所需 scale/全局归约等 kernel |
| `dequant_kernel_ms` | 反量化 kernel 时间 |
| `quant_logical_GBps`、`dequant_logical_GBps` | 逻辑内存带宽 GB/s |
| `cpu_quant_match`、`cpu_dequant_match` | 与 CPU 参考实现是否一致 |

误差比较的是原始输入与最终指定精度的反量化值：最大绝对差、绝对差均值、差值平方均值。CPU 对照检查实现正确性，与有损量化误差是两件事。

压缩率分别按有效载荷、完整文件计算。这里的时间是单次运行记录，逻辑带宽不是实测 DRAM 带宽；正式优化比较仍按 [benchmark 协议](../docs/BENCHMARK_PROTOCOL.md)执行。

---

## 生成方式

### 单次运行

```bash
python3 Core/tools/quantize.py run --input input/917/1.fp32 \
  --format mxfp8 --output-type fp16 --output-dir output/917 --save all
```

输入路径按实际文件修改。以上命令直接产生 `output/917/1_mxfp8_cuda_fp32_fp16.{lpq,fp16,json}`，不添加额外子目录。终端显示输出文件夹及各文件的绝对路径。重复运行的文件名前缀依次加 `_2`、`_3`；格式与后端包含在文件名中，避免混淆。

| 保存选项 | 保存的文件 |
|---|---|
| `--save all`（默认） | 权重、反量化张量、日志，题目要求的全部输出 |
| `--save weights` | `.lpq` 权重 |
| `--save tensor` | 指定精度的反量化张量 |
| `--save log` | `.json` 误差与性能日志 |
| `--save weights log` | 权重和日志，可自由组合三类选项 |

保存选项不改变计算范围，仍执行完整量化、反量化与 CPU 对照；未选文件仅作为临时文件处理，完成后清理。JSON 的 `packed`、`output`、`log` 在未选对应文件时为 `null`，`saved_outputs` 记录所选类型，`output_directory` 记录保存目录。终端指标始终显示。

`--output-type fp32/fp16/bf16` 控制输出精度；`--format mxfp8/nvfp4` 控制量化格式。可继续使用 `--config`，显式命令行参数覆盖配置。`--output-dir` 与旧的 `--prefix` 互斥；两者均省略时，沿用上方按输入日期/编号组织的自动目录。

### 误差评估

```bash
python3 Core/tools/quantize.py evaluate --backend cuda
```

此命令自动准备并实际读取 `input/evaluation-v1/` 的六份固定输入：三种分布 × FP32/FP16，均为 128×129、seed=1234，参数和 SHA256 见该目录的 `manifest.json`。只想生成输入时用 `python3 Core/tools/quantize.py prepare-evaluation`，无需 GPU。

输出默认保存到 `output/evaluation-v1/cuda/`，已存在则自动使用 `cuda-2/`、`cuda-3/`。MUSA 使用 `--backend musa`，读取同一批输入，结果单独保存。也可用 `--directory` 指定一个尚不存在的新目录。

**先看顶层三个分类 JSON 或 `summary.json`，需要某组二进制文件时再进入分布/格式目录。** 每个分类 JSON 有 48 组记录，总计 144 组：3 分布 × 2 输入类型 × 2 格式 × 2 缩放方式 × 2 舍入方式 × 3 输出类型。这是项目的配置覆盖范围，题目没有要求必须生成 144 组。

每组的 `.lpq`、反量化张量及 `.json` 都持久保存，无需手动复制。每条日志中的 `input`、`packed`、`output` 指向实际文件。`status.json` 的 `state=complete` 且 `completed=144` 表示整批运行完成；`failed` 或 `running` 只表示部分结果，不能当作全量通过。

---

## 清理说明

`output/` 下的旧结果可以清理，**但不要删除 `input/benchmark-v1/` 与 `records/` 的原始证据**。

二进制 `.fp32`、`.fp16`、`.bf16`、`.lpq` 被 `.gitignore` 排除，需在本机运行命令生成；JSON 和文档可纳入版本库。目录存在不代表里面的二进制数据已经生成。

---

## 相关文档

- 目录格式规范：[OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)
- 输入文件格式：[input/README.md](../input/README.md)
- 过程证据：[records/README.md](../records/README.md)
