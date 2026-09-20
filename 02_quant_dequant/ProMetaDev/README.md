# MXFP8 / NVFP4 低精度模拟与反量化（CUDA）

2026 夏季训练营 CUDA 方向项目阶段 —— 选题二。

在**普通 CUDA GPU** 上以纯软件方式实现 MXFP8 / NVFP4 低精度浮点格式的
编码、打包、缩放、解包、反量化与误差评估，**不依赖 Hopper / Blackwell / Ada
的任何新硬件指令，也不要求硬件原生支持 FP8 / FP4 Tensor Core**。

## 1. 支持的格式与功能

| 项目 | 说明 |
|---|---|
| 元素格式 | MXFP8：E4M3 / E5M2；NVFP4：E2M1（FP4） |
| 共享缩放 | MXFP8：E8M0（2 的整数次幂），默认 block = 32 |
| 两级缩放 | NVFP4：block 局部 E4M3 缩放 + 张量级 FP32 全局缩放，默认 block = 16 |
| 缩放模式 | `tensor`（整张量一个缩放因子）/ `block`（按 block 切分） |
| block 粒度 | 16 / 32 走共享内存优化路径；其他任意正整数走通用路径（已实测 64 / 128） |
| 舍入模式 | `nearest`（最近偶数）/ `stochastic`（随机舍入） |
| 反量化输出 | fp16 / bf16 / fp32 |
| 4bit 打包 | 每字节存 2 个元素（高位/低位各一个 nibble）；反量化用 `uchar4` 向量化 packed load |
| 误差指标 | max_abs / MAE / MSE / RMSE / NMAE / SQNR，分「纯量化误差(fp32 域)」与「端到端误差」两个口径 |
| 输入类型 | fp32 / fp16 |

## 2. 目录结构

```
.
├── CMakeLists.txt
├── build.sh                 # 一键构建（自动探测 CUDA 环境）
├── .clang-format            # 代码格式化配置
├── .gitignore               # 排除构建产物与大体积运行数据
├── include/
│   ├── lowp.h               # 编解码核心（E4M3/E5M2/E2M1/E8M0，host+device 共用）
│   ├── io.h                 # 张量/配置/压缩权重文件 I/O
│   ├── metrics.h            # 误差指标与带宽统计
│   └── quant.h              # 量化/反量化流程接口
├── src/
│   ├── main.cpp             # CLI：gen / quant / dequant / selftest
│   ├── io.cpp               # I/O 实现（含 host 端 fp16/bf16 软件转换）
│   ├── metrics.cpp          # 误差指标实现
│   ├── reference.cpp        # CPU 标量参考实现（正确性基准）+ fp32 域反量化
│   └── kernels.cu           # CUDA kernel 与 GPU 流程编排
├── tools/
│   └── cross_check.py       # 第三方标准实现（ml_dtypes）交叉验证
├── configs/                 # 11 组量化配置
├── tests/run_tests.sh       # 端到端测试脚本（39 项）
└── report.md                # 总结报告
```

## 3. 构建

```bash
bash build.sh          # 默认目标架构 sm_90，可传参覆盖：bash build.sh 89
```

构建脚本会自动探测 `/usr/local/cuda*` 并配置 `PATH` / `CUDACXX`。

## 4. 使用

```bash
# 0) 编解码自检
./build/lowprec selftest

# 1) 生成测试张量（kind = random | normal | outlier | zeros）
./build/lowprec gen normal 4096 4096 outputs/data/normal.txt
./build/lowprec gen outlier 1024 1024 outputs/data/outlier.txt --fp16

# 2) 量化 + 反量化 + 误差/性能日志
./build/lowprec quant outputs/data/normal.txt configs/mxfp8_block_fp16.cfg outputs/results/mxfp8
./build/lowprec quant outputs/data/normal.txt configs/nvfp4_block_fp16.cfg outputs/results/nvfp4 --no-cpu

# 3) 从已保存的低精度权重文件重新反量化
./build/lowprec dequant outputs/results/nvfp4.quant outputs/results/nvfp4.reload.bin

# 4) 第三方标准实现交叉验证（需 ml_dtypes）
python3 tools/cross_check.py outputs/data/normal.txt outputs/results/nvfp4.quant

# 5) 端到端测试（3 种分布 × 多配置 + 边界 + 交叉验证，共 39 项）
bash tests/run_tests.sh
```

## 5. 输入文件格式

**张量数据文件**（文本头 + 二进制数据段，行主序）：

```
[header]
num_rows: 4096
num_cols: 4096
dtype: fp32            # fp32 或 fp16

[data]
<num_rows * num_cols 个元素，按 dtype 紧密排列的二进制数据>
```

**量化参数文件**（`key = value`，支持 `#` 注释）：

```
format = "mxfp8"        # mxfp8 / nvfp4
elem_format = "e4m3"    # e4m3 / e5m2（仅 mxfp8）
block_size = 32         # mxfp8 默认 32，nvfp4 默认 16
scale_mode = "block"    # tensor / block
output_type = "fp16"    # fp16 / bf16 / fp32
rounding = "nearest"    # nearest / stochastic
target_gpu = "T4"
seed = 1234
```

## 6. 输出文件格式

**低精度权重文件**（`.quant`，二进制，小端）：

```
偏移  大小            内容
0     4               magic = "LPQ1"
4     40              头部：format/elem/rows/cols/block_size/scale_mode/
                      out_dtype/round_mode/num_blocks/global_scale/
                      packed_bytes/scale_bytes
44    packed_bytes   打包后的低精度数据（4bit 为每字节 2 元素）
44+   scale_bytes    缩放因子数组（MXFP8 为 E8M0；NVFP4 为每 block 一个 E4M3）
```

**反量化张量**（`.dequant.bin`）：裸二进制、行主序，类型由 `output_type` 指定。

**误差与性能日志**（`.log`）：最大绝对误差、MAE、MSE、RMSE、压缩率、
量化/反量化 kernel 时间、有效内存带宽、CPU 参考耗时与加速比、
以及 CPU/GPU 一致性与权重文件可重载校验结果。

## 7. 数值定义摘要

- **MXFP8**：`x ≈ fp8(x / 2^e) * 2^e`，`e = ceil(log2(amax_block / max_elem))`，
  以 E8M0 存储；E4M3 的 `max_elem = 448`，E5M2 为 `57344`。
- **NVFP4**：`x ≈ fp4(x / (s_b * s_g)) * s_b * s_g`，
  其中 `s_g = gmax / (6 * 448)` 为张量级 FP32 全局缩放，
  `s_b` 为 block 内 E4M3 局部缩放，`s_b * s_g` 使 block 内最大元素恰好落到 FP4 上限 6。
- **E2M1 可表示幅值**：`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`。

详见 [report.md](report.md)。
