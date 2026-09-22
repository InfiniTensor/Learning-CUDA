# MXFP8 / NVFP4 量化与反量化

作者：0xtoward。总结见 [中文报告](REPORT.md)，完整数据见 [误差表](accuracy.md)、[性能表](benchmark.md) 和 `results/`。

2026-09-22 补记：[Marlin 源码学习、CUDA Core 实验与 TODO](MARLIN_LEARNING.md)，记录寄存器复用、向量读取、异步复制和 warp 分工的实际迭代。

## 快速运行

环境：Linux/WSL、Python 3.11+、CUDA Toolkit 12.x、CUDA 版 PyTorch。基础 CUDA core 编译 SM75/SM89，使用普通 CUDA 指令；本机验证为 RTX 4070 Laptop（SM89）。

在本目录执行：

```bash
python -m pip install -r requirements.txt
make -j2
python test_codec.py
python -c "import torch; from tensor_io import save_tensor; save_tensor('input.lpt', torch.randn(512,1024,dtype=torch.float16))"
python run_codec.py --input input.lpt --config configs/mxfp8.toml --output-dir output/mx
python run_codec.py --input input.lpt --config configs/nvfp4.toml --output-dir output/nv
python run_codec.py --input input.lpt --config configs/nvfp4_4over6.toml --output-dir output/nv46
```

每次输出 `weights.lpq`（packed data + scales）、`dequantized.lpt`、`metrics.json`。日志包含 MaxAbs、MAE、MSE、相对 L2、压缩率、quant/dequant 时间、有效带宽；NVFP4 全局归约单独计时。

| 配置 | 可选值 |
|---|---|
| format / block_size | mxfp8 / 32；nvfp4 / 16 |
| scale_mode | block、tensor |
| output_type | fp16、bf16、fp32 |
| rounding | nearest（最近偶数）、stochastic（seed 可复现） |
| scale_policy | floor、rceil（MXFP8） |
| four_over_six | true：NVFP4、block、nearest 的自适应 4/6 缩放 |
| target_gpu | 报告标签，不改变运行设备 |

文件输入为有限 FP32/FP16 矩阵，Python API 另支持 BF16。文件布局为 8 字节 magic、uint32 little-endian JSON 长度、UTF-8 header、行主序二进制 payload。header 包含 `num_rows`、`num_cols`、`dtype`；量化 header 另包含格式、选项及各 payload 段长度。

## 可选对照、官方 oracle 与模型接入

```bash
python -m pip install -r requirements-extra.txt
make extras -j2
python benchmark_variants.py
# 下载模型后指定本地 safetensors 文件：
export QWEN_SMALL_CHECKPOINT=/path/to/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors
python evaluate_accuracy.py
python export_tables.py
python run_smoke.py --model /path/to/Qwen3.5-0.8B
```

`benchmark_timer.py` 随项目提供，无个人 Skill 路径依赖。Triton 快速转换对照运行于 SM89+。

官方 4over6 reference 固定版本，禁用其原生 CUDA 扩展：

```bash
git clone https://github.com/mit-han-lab/fouroversix.git /path/to/fouroversix
git -C /path/to/fouroversix checkout dadfad6901d473a734fe71e0b082e70ee993e23a
SKIP_CUDA_BUILD=1 python -m pip install --no-deps --no-build-isolation -e /path/to/fouroversix
export FOUROVERSIX_REPO=/path/to/fouroversix
python test_codec.py --oracle
python test_extended.py
compute-sanitizer --tool memcheck --error-exitcode 1 python sanitizer_smoke.py
```

`test_extended.py` 使用 Triton、官方 oracle 和上述真实权重。普通 `test_codec.py` 无需下载模型或 oracle。服务为本机 `127.0.0.1` HTTP 接口：低精度权重驻留，每层调用自己的 CUDA dequant，再用 PyTorch `F.linear`。原始回答、显存、延迟见 `results/smoke_small.json`。

## 文件索引

| 文件 | 职责 |
|---|---|
| codec.cu、formats.cuh | 软件编解码、缩放、packed load/store、dtype 转换 |
| codec.py | 当前 CUDA stream 上的 C ABI 封装 |
| tensor_io.py、run_codec.py、configs/ | 文件和参数接口 |
| reference.py、test_*.py | 独立 NumPy 参考和测试，运行时与测试分离 |
| mxfp8_native_minimal.cu、triton_codec.py | CUDA Math API、Triton 可选对照 |
| smoke_server.py、run_smoke.py | 真实模型接入与固定答案回归 |
| REPORT.md、results/ | 中文总结、Marlin 学习记录、实测证据与 TODO |

测试结果和性能表记录于 2026-09-20；Marlin 优化记录来自此前同机实验。模型权重、编译产物和大型 profiler trace 不纳入源码提交。
