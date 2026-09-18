# 生成实验矩阵与端到端批量验收

本项目把“生成数据”“运行完整 app”“位级单元测试”分成三个层次：

- `scripts/generate_tensor.py`：生成合法的输入用途 QDTENSOR；
- `scripts/run_e2e_suite.py`：批量调用真实 CUDA app，检查产物与 JSON；
- `tests/`：通过 CTest 验证 codec、I/O、CPU reference 与 CUDA kernel 的细粒度正确性。

脚本不是 CTest 的替代品。它们覆盖真实命令行、文件路径、QDWGT 写读、反量化输出
写读和 JSON 指标边界；位编码或单元素舍入错误仍应优先在 CTest 中定位。

## 生成单份输入矩阵

生成器只使用 Python 标准库，不需要 NumPy、CUDA 或 GPU。它写出的文件固定为
QDTENSOR v1、little-endian、`tensor_role = input`，物理输入 dtype 可为 FP16 或
FP32。

```shell
cd "02_quant_dequant/黄新颖"

# 均匀分布 FP32，适合基本量化路径。
python3 scripts/generate_tensor.py \
  --output outputs/inputs/uniform_512x513_fp32.qdtensor \
  --rows 512 --cols 513 --dtype fp32 \
  --distribution uniform --seed 20260917

# 标准正态分布 FP16，生成器会先采样再真实窄化为 binary16 payload。
python3 scripts/generate_tensor.py \
  --output outputs/inputs/normal_512x513_fp16.qdtensor \
  --rows 512 --cols 513 --dtype fp16 \
  --distribution normal --seed 20260918

# 稀疏异常值：99% 附近为 N(0, 1)，约 1% 元素为 ±32。
python3 scripts/generate_tensor.py \
  --output outputs/inputs/outlier_512x513_fp32.qdtensor \
  --rows 512 --cols 513 --dtype fp32 \
  --distribution outlier --seed 20260919 \
  --outlier-probability 0.01 --outlier-magnitude 32
```

生成成功时 stdout 输出一行 JSON，其中包含 shape、dtype、seed、元素数和真实文件
字节数。相同的全部参数必定产生逐字节相同的输入文件。

### 三种分布的含义

| `--distribution` | 生成规则 | 主要观察点 |
| --- | --- | --- |
| `uniform` | 默认从 $[-1, 1]$ 均匀采样。 | 基本 scale、pack/unpack 与平均误差。 |
| `normal` | 默认从 $mathcal{N}(0, 1)$ 采样。 | 常规近零数值与格式 subnormal/normal 区域。 |
| `outlier` | 基础为 $mathcal{N}(0, 1)$，默认 1% 以概率替换为 $pm32$。 | 异常值如何拉大 block/tensor 的 scale，并影响小值误差。 |

`--uniform-low`、`--uniform-high`、`--normal-mean`、`--normal-stddev`、
`--outlier-probability` 和 `--outlier-magnitude` 都可调整。所有参数和值在写入前会
检查有限性；若 FP16 无法表示某个生成值，脚本会明确失败而不是静默写入 Inf。

## 批量端到端 suite

先构建 release app。性能数据应该来自 release，而不是带调试信息的 debug：

```shell
cmake --preset rtx4060-release
cmake --build --preset rtx4060-release -j
```

最小 smoke suite 使用较小矩阵、一个真实 TOML、一个输入 dtype、一种分布和一轮正式
运行，不做 warmup；适合确认命令行和 GPU 环境可用：

```shell
python3 scripts/run_e2e_suite.py \
  --rows 128 --cols 129 \
  --input-dtypes fp32 \
  --distributions uniform \
  --config-names mxfp8_block_nearest_fp16 \
  --warmups 0 --repeats 1 \
  --output-dir outputs/e2e_smoke
```

`configs/` 中提交的是 **真实 app 配置**，而不是脚本临时写出的配置：

- MXFP8：block/tensor、nearest/stochastic、FP16/BF16/FP32 输出，共 12 份；
- NVFP4：只允许 block、nearest/stochastic、FP16/BF16/FP32 输出，共 6 份；
- stochastic TOML 均显式写入 `stochastic_seed = 20260917`；nearest TOML 使用默认的
  seed 0。

默认 suite 使用三种分布、两种物理输入 dtype、全部 18 份配置、`1024 × 1025` 矩阵、
零次额外功能运行和一次正式运行，因此会进行
$3 \times 2 \times 18 \times (0 + 1) = 108$ 次完整 app 调用：

```shell
python3 scripts/run_e2e_suite.py \
  --output-dir outputs/e2e_release
```

它会创建如下结构：

```text
outputs/e2e_release/
├── inputs/                 # 六份固定 seed 的 QDTENSOR：3 分布 × 2 输入 dtype
├── runs/
│   └── <input_dtype>/<distribution>/<config_name>/<warmup_or_repeat>/
│       ├── weights.qdwgt
│       ├── dequantized.qdtensor
│       └── report.json
└── summary.md              # 正式 repeat 的均值、中位数和误差汇总
```

若只希望跑某一格式或某一输入类型，可以缩小组合：

```shell
# 仅 MXFP8 tensor/stochastic/BF16，FP32 正态输入；五轮正式重复用于观察波动。
python3 scripts/run_e2e_suite.py \
  --config-names mxfp8_tensor_stochastic_bf16 \
  --input-dtypes fp32 \
  --distributions normal \
  --rows 2048 --cols 2049 \
  --warmups 2 --repeats 5 \
  --output-dir outputs/mxfp8_bf16_release

# 仅针对异常值，检查 NVFP4 分层缩放的两种舍入和全部反量化输出类型。
python3 scripts/run_e2e_suite.py \
  --formats nvfp4 \
  --distributions outlier \
  --input-dtypes fp16 fp32 \
  --output-dir outputs/nvfp4_outlier_release
```

## suite 校验的内容

每次 app 调用成功后，脚本读取 `report.json`，并与命令行规格、真实文件系统状态做
交叉验证：

- 输入 shape、输入 dtype、format、block size、scale mode、rounding、stochastic seed、
  输出 dtype 和 `target_gpu` 必须与本次真实 TOML 一致；
- MXFP8 payload 必须是一个元素一字节，NVFP4 必须是两个元素一字节；
- local scale 数量必须等于
  $\mathrm{rows} \times \lceil \mathrm{cols}/\mathrm{block\_size} \rceil$；
- NVFP4 的 `global_scale_bytes` 必须为 4，MXFP8 必须为 0；
- JSON 声明的 QDWGT、反量化 QDTENSOR 文件大小必须等于实际文件大小；
- 压缩率、逻辑字节数、误差、kernel 时间与带宽必须满足 metrics 的公式和有限性约束；
- CUDA Event 得到 `0.0 ms` 时，带宽必须为 `null`；正时间时带宽必须为有限正数。

因此 suite 既能发现 app 失败，也能发现“app 虽然退出为 0、但产物/报告字段不自洽”的
问题。

## 性能解释与独立 benchmark

`summary.md` 汇总的是 app 返回的 CUDA Event kernel 时间，H2D、D2H、文件 I/O、
host 分配、CUDA context 初始化均不在其中。默认全量 108 次运行的主要用途是功能覆盖：
它会确认真实 QDTENSOR、QDWGT、反量化文件和 JSON 报告彼此自洽。

`run_e2e_suite.py` 的一次 warmup/repeat 都会启动一个新的 `quant_dequant` 进程。因此
它的 `--warmups` 只能表示“额外的端到端功能验证次数”，**不能**预热下一次 repeat 的
CUDA stream、上下文或 device allocation。脚本在 `--warmups > 0` 时会打印这一提醒；
默认仍保持 `warmups = 0`、`repeats = 1`，避免将全配置功能验收误解为性能 benchmark。

正式 kernel 性能使用 `benchmarks/quant_dequant_bench`。它在**同一个进程**中只读取一次
QDTENSOR 与 TOML，先执行不计入统计的 warmup，再执行正式 repeat，并报告 quant/dequant
的 min、mean、median、p95、max 及由 mean 推导的逻辑有效带宽：

```shell
cd "02_quant_dequant/黄新颖"
cmake --build --preset rtx4060-release --target quant_dequant_bench -j

# 建议先用较大且列数不整除 block_size 的 shape，例如 4096 x 4097。
./build/rtx4060-release/benchmarks/quant_dequant_bench \
  --input outputs/inputs/normal_4096x4097_fp32.qdtensor \
  --config configs/mxfp8_block_nearest_fp32.toml \
  --warmups 10 --repeats 30 \
  --output outputs/benchmarks/mxfp8_block_normal_fp32.json
```

benchmark 每轮仍通过公共 profile pipeline 执行完整的 H2D、CUDA kernel 和 D2H，因而
数值路径与 app 一致；其 CUDA Event 仍只测 kernel。当前 pipeline 会在每轮管理自己的
stream 和 `thrust::device_vector`，但这些分配、拷贝和 I/O 都位于 Event 区间外。于是该
工具消除了“每次性能样本都重启进程”的问题，却不把结果伪装成“已复用 device buffer 的
极限吞吐”。后者可作为下一阶段专项优化。

### 批量性能配置遍历

`scripts/run_benchmark_suite.py` 负责自动生成一份输入、扫描 `configs/` 中真实 TOML、
逐项调用 `quant_dequant_bench`，并校验每个 JSON 的输入、配置、warmup/repeat、分位数
顺序与带宽字段。它不会写 QDWGT/QDTENSOR 中间产物；文件级正确性仍由
`run_e2e_suite.py` 负责。

默认性能组合选择 `4096 × 4097`、FP32 normal 输入、`warmups = 10`、`repeats = 30`，
并只筛选 `output_type = fp32` 的六份 TOML。原因是当前反量化 kernel 先统一产生 FP32，
FP16/BF16 的物理窄化在随后 QDTENSOR I/O 层完成；把三种输出 TOML 都作为 kernel
benchmark 样本会造成重复工作。默认六份分别覆盖 MXFP8 的 block/tensor、nearest/
stochastic，以及 NVFP4 的 block、nearest/stochastic。

```shell
cd "02_quant_dequant/黄新颖"
cmake --build --preset rtx4060-release --target quant_dequant_bench -j

# 默认性能组合：6 份真实 configs/ TOML。
python3 scripts/run_benchmark_suite.py \
  --output-dir outputs/benchmark_suite

# 若要逐项性能记录全部 18 份提交的配置文件：
python3 scripts/run_benchmark_suite.py \
  --output-dtypes fp16 bf16 fp32 \
  --output-dir outputs/benchmark_suite_all_configs
```

输出结构为：

```text
outputs/benchmark_suite/
├── inputs/
│   └── normal_fp32_4096x4097_seed20260917.qdtensor
├── runs/
│   └── <config_name>/benchmark.json
└── summary.md
```

若要从空的 `outputs/` 重新执行完整验收，先运行 `run_e2e_suite.py` 生成/验证全部文件
路径，再运行上述 performance suite。前者默认 108 次完整 app 调用，后者默认 6 个
同进程 benchmark；二者的报告不可混为同一统计口径。
