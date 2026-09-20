# Hadamard 变换加速（CUDA）

2026 夏季训练营 CUDA 方向项目阶段 —— 选题三。

实现快速 Walsh-Hadamard 变换（FHT）的 CUDA kernel，并把它与 FP8 量化**融合**成单个
kernel（中间结果只留在寄存器，不落显存），用于 QuaRot / SpinQuant 这类"旋转后量化"
的预处理场景。

## 1. 功能与要求对照

| 题目要求 | 实现情况 |
|---|---|
| 读取输入数据（FP32/FP16/BF16） | 三种输入精度均支持（host 端软件位运算转换） |
| FHT 支持 n = 64 / 128 / 256 | 全部支持（另有 32 / 512，按 2 的幂分派） |
| 支持批量多行、逐行独立变换 | 每个 warp 以 grid-stride 处理多行 |
| 与量化融合（FHT → 量化 → 输出） | **融合 kernel**：蝶形在寄存器内完成，随即 warp 归约出块 amax 并量化输出 |
| 误差 ≤ 1e-3（与参考实现比对） | 与"直接构造 H_n 做矩阵乘"的参考对比：**max_abs_error ≈ 7.6e-06** |
| 不依赖 Hopper/Blackwell/Ada 新特性 | 未使用 Tensor Core、未使用 `cuda_fp8.h`，纯位运算 + warp shuffle |
| 性能日志：FHT / 量化 / 融合 时间与带宽 | 全部输出，另含融合加速比与相对基线加速比 |
| 平台适配 | 默认英伟达；无架构特性依赖 |

## 2. 目录结构

```
.
├── CMakeLists.txt
├── build.sh                  # 一键构建（自动探测 CUDA 环境）
├── .clang-format / .gitignore
├── include/
│   ├── types.h               # 配置、张量、量化结果、性能统计
│   ├── fp8.h                 # E4M3 / E5M2 / E8M0 纯软件编解码（host+device 共用）
│   ├── io.h                  # 数据/参数/输出文件读写与误差统计
│   └── hadamard.h            # CPU 参考与 GPU 接口
├── src/
│   ├── main.cpp              # CLI: gen / run / quant / verify / selftest
│   ├── io.cpp                # I/O 实现（含 fp16 / bf16 软件转换）
│   ├── cpu_ref.cpp           # CPU 参考：FWHT(O(n log n)) + 定义式矩阵乘(O(n^2))
│   └── fht.cu                # FHT kernel、融合 kernel、朴素矩阵乘基线
├── tools/torch_ref.py        # PyTorch/NumPy 独立参考 + 矩阵乘性能基线
├── configs/                  # 5 组配置（n、scale、量化格式）
├── tests/run_tests.sh        # 端到端测试（13 项）
└── report.md                 # 总结报告
```

## 3. 构建

```bash
bash build.sh          # 默认 sm_90，可传参覆盖：bash build.sh 89
```

## 4. 使用

```bash
# 0) 自检
./build/hadamard selftest

# 1) 生成数据（kind = uniform | normal | outlier）
./build/hadamard gen normal 262144 128 outputs/data/base.txt --seed 7
./build/hadamard gen outlier 262144 128 outputs/data/outlier.txt --dtype fp16

# 2) 生成 PyTorch/NumPy 参考结果并测得矩阵乘基线时间
python3 tools/torch_ref.py outputs/data/base.txt outputs/ref.txt --n 128 --scale 1.0

# 3) 运行：FHT +（可选）量化，输出结果与性能日志
./build/hadamard run outputs/data/base.txt configs/fht_128_quant_e4m3.cfg \
                   outputs/results/run \
                   --ref outputs/ref.txt --torch-ms 1.14

# 4) 仅量化（不旋转），用于对比旋转对量化误差的影响
./build/hadamard quant outputs/data/outlier.txt configs/fht_128_quant_e4m3.cfg \
                     outputs/results/raw

# 5) 单独比对两个张量文件
./build/hadamard verify outputs/ref.txt outputs/results/run.fht.bin

# 6) 端到端测试
bash tests/run_tests.sh
```

## 5. 文件格式

**输入数据文件**（文本头 + 二进制数据段）：

```
[header]
rows: 262144
cols: 128
layout: row_major
dtype: fp32          # fp32 | fp16 | bf16
apply_dim: last

[data]
<rows*cols 个元素，行主序，按 dtype 紧密排列>
```

**参数文件**（`key = value`）：

```
hadamard_size = 128        # n ∈ {64, 128, 256}
scale = 1.0                # 输出乘性缩放，如 1/sqrt(n)
quantize_enable = true
quant_format = "fp8_e4m3"  # fp8_e4m3 | fp8_e5m2
block_size = 32            # 量化块大小
```

**输出**：

- `<prefix>.fht.bin`：变换后的张量（与输入同格式，默认 fp32 写出）
- `<prefix>.quant.bin`：量化结果（`HQ01` 头 + FP8 数据 + E8M0 缩放因子）
- `<prefix>.log`：性能与误差日志

## 6. 算法要点

**FHT（warp-per-row）**：一行 n 个元素由 32 个 lane 共同处理，lane `l` 持有元素
`{ l + 32k : k = 0..V-1 }`，`V = n/32`。元素下标 `i = l + 32k`，因此 bit0..4 由 lane 决定、
bit5.. 由 k 决定，于是 log2(n) 个蝶形阶段被拆成：

- 阶段 0..4：跨 lane，用 `__shfl_xor_sync`；
- 阶段 5..：lane 内，在寄存器数组的不同 slot 间完成。

载入/写回按 lane 连续访问（lane 读 `base + lane + 32k`），访存完全合并；整个变换
只占 30 个寄存器、零 spill。

**蝶形符号约定**（与定义式 `H[i][j] = (-1)^popcount(i&j)` 对齐）：设 a 为下标 bit=0 的
元素、b 为 bit=1 的元素，则 bit=0 侧得 `a+b`、bit=1 侧得 `a-b`。

**融合**：由于 `block_size = 32` 恰好等于一个 slot（元素 `[32k, 32k+32)`），
FHT 完成后每个 slot 的 amax 只需一次 warp 归约即可得到，随后立即量化并写出。
整个"变换 + 求缩放 + 量化"全部在寄存器内完成，相比"先写 FHT 结果再读回来量化"
省掉一整趟显存往返。

详见 [report.md](report.md)。
