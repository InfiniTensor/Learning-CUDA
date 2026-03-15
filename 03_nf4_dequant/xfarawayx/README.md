# NF4 反量化 CUDA Kernel

NF4（Normal Float 4）双重量化权重的 GPU 反量化实现，兼容 bitsandbytes 格式。技术细节与实验结果见 [`docs/report.md`](docs/report.md)。

## 目录结构

```
03_nf4_dequant/
├── run.sh                        # 统一入口脚本
├── kernel/
│   ├── CMakeLists.txt            # CMake 构建 (自动检测 GPU 架构)
│   ├── main.cu                   # 主程序: 文件 IO、kernel 启动、性能计时
│   └── nf4_dequant_kernel.cuh    # 反量化 kernel 实现
├── kernel_noncuda/               # 国产 GPU 适配
│   ├── iluvatar/                 # 天数智芯 (clang++ / CUDA 兼容)
│   ├── moore/                    # 摩尔线程 (mcc / MUSA)
│   └── mutex/                    # 沐曦 (mxcc / MACA)
├── scripts/
│   ├── generate_data.py          # 用 bitsandbytes 生成 NF4 量化数据 + 参考输出
│   ├── verify.py                 # 正确性验证: CUDA 输出 vs bitsandbytes 参考
│   └── bench_bnb.py              # bitsandbytes 性能基准
├── docs/
│   └── report.md                 # 实现报告
└── data/                         # 生成的数据 (自动创建)
```

## 快速开始

```bash
# 全流程: 生成数据 → 编译 → 运行 kernel → 验证正确性 → bnb 性能对比
./run.sh all
./run.sh            # 等价于 ./run.sh test
```

## 子命令与选项

| 命令 | 说明 |
|------|------|
| `./run.sh generate` | 仅生成 NF4 量化测试数据 |
| `./run.sh build` | 仅编译 CUDA kernel |
| `./run.sh test` | 生成数据 → 编译 → 运行 → 验证正确性 (默认) |
| `./run.sh bench` | bitsandbytes 基准性能测试 |
| `./run.sh all` | 完整流程 |

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--rows` | 4096 | 矩阵行数 |
| `--cols` | 4096 | 矩阵列数 |
| `--blocksize` | 64 | 量化块大小 (64/128/256/…/4096) |
| `--compute_type` | bf16 | 输出类型 (bf16/fp16) |
| `--seed` | 42 | 随机种子 |
| `--gpu_arch` | 自动检测 | GPU 架构, 如 80(A100)/89(4090)/90(H100) |
| `--warmup` | 10 | 预热次数 |
| `--repeats` | 100 | 计时重复次数 |
| `--sweep` | - | bench 时扫描多种矩阵大小 |

```bash
# 示例
./run.sh --rows 4096 --cols 11008 --blocksize 128
./run.sh --compute_type fp16
./run.sh bench --sweep
```

## 依赖

- CUDA Toolkit
- Python 3.8+, PyTorch (CUDA), bitsandbytes >= 0.43, NumPy
