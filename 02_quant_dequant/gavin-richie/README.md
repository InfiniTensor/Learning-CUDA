# MXFP8 / NVFP4 低精度模拟与反量化（CUDA 软件实现）

不依赖任何 FP8/FP4 硬件指令（兼容 Maxwell 及以上普通 CUDA GPU）的低精度格式
软件模拟：量化、4-bit 打包、缩放、解包、反量化与误差评估。

## 构建

需要 CUDA Toolkit >= 11.5（建议 12.x）、CMake >= 3.18。

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # 4060/4090(D) (Ada)
cmake --build build -j
cd build && ctest          # 单元测试 + GPU roundtrip 测试
```

架构对照：3090=86，4060/4090=89，H20=90，5090=120，Hopper=90，
数据中心 Blackwell=100/101/103，消费级 Blackwell=120。
默认 `CMAKE_CUDA_ARCHITECTURES=89`（已在 RTX 4060 / RTX 4090 D /
RTX 5090 三档上完成复测，误差结果逐位一致）。

## 运行

```bash
# 生成测试张量（random / normal / outlier 分布，fp32 或 fp16）
python3 scripts/gen_tensor.py artifacts/t.bin 4096 4096 fp32 normal

# 端到端：量化 + 反量化 + 误差/性能日志
./build/lowp run --input artifacts/t.bin --config q.txt \
    --out-weight w.bin --out-dq out.bin --log log.txt --dist normal

# 或分步：先量化保存权重，再从权重文件反量化
./build/lowp quant   --input artifacts/t.bin --config q.txt --out-weight w.bin
./build/lowp dequant --weight w.bin --config q.txt --out-dq out.bin
```

量化参数文件（`q.txt`，python 风格键值对）：

```python
format = "mxfp8"      # mxfp8 (E4M3) / mxfp8-e5m2 / nvfp4
block_size = 32       # mxfp8 默认 32, nvfp4 默认 16
scale_mode = "block"  # tensor / block
output_type = "fp16"  # fp16 / bf16 / fp32
rounding = "nearest"  # nearest / stochastic
target_gpu = "RTX 4060"
```

## 文件格式

- 张量文件：`int64 rows; int64 cols; char dtype[4] ("fp32"/"fp16"); 行主序数据`
- 低精度权重文件：`WeightFileHeader`（magic "QWHT"，见
  `include/quant_types.h`）+ packed data + scale 数组 + fp32 scale。
  NVFP4 严格 2 元素/字节（元素 2k 在低半字节）。
- 误差/性能日志：max abs error、MAE、MSE、压缩率、量化/反量化 kernel
  时间（cudaEvent）、有效内存带宽（GB/s）。

## Profiling（ncu / nsys）

```bash
scripts/profile.sh ./build/lowp artifacts/t.bin q.txt artifacts/prof
```

生成的 `.nsys-rep` / `.ncu-rep` 是跨平台文件，可直接在 Windows 宿主机上用
Nsight Systems / Nsight Compute GUI 打开（GUI 版本需 >= 采集端 CLI 版本）。
注意：WSL2 与未授权容器中 ncu 无法访问 GPU 性能计数器（`ERR_NVGPUCTRPERM`，
容器场景通常是宿主机驱动 `RmProfilingAdminOnly=1` 且未授予
`CAP_SYS_ADMIN`），脚本会识别为"跳过"并记录日志；nsys 时间线不受影响。

## GPU 环境排障（共享 GPU 容器）

容器里的 GPU 访问有三种已知故障形态（报错形态各异、根因不同）：

```bash
bash scripts/gpu_env_doctor.sh        # 体检：exit 0=健康 1=不可用 2=可用但有警告
bash scripts/gpu_env_doctor.sh --fix  # 对可修复项输出 eval 可用的修复命令
```

| 形态 | 现象 | 根因 |
|---|---|---|
| 设备节点缺失 | `cuInit` 返回 100 / "no NVIDIA driver" | 宿主机未注入或已回收 `/dev/nvidia[0-9]*`、`/dev/nvidia-uvm` |
| 驱动库 stub | `dlopen: file too short` | `libcuda.so.*` / `libnvidia-ml.so.*` 是 0 字节占位文件 |
| 版本混载 | `cuInit` 返回 **803** | `LD_LIBRARY_PATH` 混入旧 cuda-compat 库（如 570）与内核驱动（如 610）组合不受支持 |

对形态 2/3，doctor 的 `--fix` 会给出把相应目录从 `LD_LIBRARY_PATH`
移除的命令；形态 1 只能由宿主机重新注入（重启容器或
nvidia-container-cli）。`te_compare.py` 启动时自带预检，失败时直接
打印上述根因提示。

## Transformer Engine 对比（可选）

```bash
python3 scripts/te_compare.py artifacts/t4096_normal.bin --dist normal \
    --out artifacts/log_te_e4m3_normal.txt

# TE >= 2.0 + Blackwell：MXFP8 / NVFP4 block-scaling recipe 对照
python3 scripts/te_compare.py artifacts/t4096_normal.bin --format mxfp8 \
    --dist normal --out artifacts/log_te_mxfp8_normal.txt
python3 scripts/te_compare.py artifacts/t4096_normal.bin --format nvfp4 \
    --nvfp4-variant vanilla --dist normal \
    --out artifacts/log_te_nvfp4_normal.txt
```

`--format e4m3|e5m2` 走 per-tensor FP8 cast（TE 1.x 与 TE 2.x 均可用）；
`--format mxfp8|nvfp4` 走 TE 2.x 的 `MXFP8Quantizer` /
`NVFP4Quantizer`（block-scaling recipe），需 TE ≥ 2.0 且硬件支持。
NVFP4 提供 `vanilla`（与软件路径直接可比：无 RHT / stochastic rounding
/ 2D）与 `full`（recipe 默认含随机 Hadamard 变换）两个变体。
所有格式输出与 `lowp run --log` 同口径的误差/带宽日志。

## CUDA Math API 一致性测试（可选）

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DENABLE_NATIVE_CVT_TEST=ON   # 自动加 120a 激活原生 cvt 指令
cmake --build build -j
cd build && ctest
```

`test_native_cvt` 在 host 与 device 上穷举所有 FP8/E2M1 码空间，并在
device 上扫描 4M 随机 32-bit 值（含 NaN/inf/denorm），验证软件
`include/fp_formats.h` 与 `__nv_cvt_float_to_fp8` / `__nv_cvt_fp4x2_*`
硬件指令逐位一致（4.5 节）。需要 CUDA ≥ 12.8（`cuda_fp4.h`）。

## 目录

- `include/fp_formats.h` — E4M3/E5M2/E2M1/E8M0 软件编解码（host/device）
- `src/quant_kernels.cu` — 量化/反量化/误差统计 CUDA kernels
- `src/file_io.cpp` — 张量/参数/权重文件 IO
- `src/main.cu` — CLI
- `tests/` — 编解码单元测试、文件 IO 测试、GPU roundtrip 测试
- `scripts/te_compare.py` — Transformer Engine per-tensor / MXFP8 /
  NVFP4 对比（带 GPU 环境预检）
- `scripts/gpu_env_doctor.sh` — 共享容器 GPU 环境体检与修复提示
- `artifacts/` — 样例日志与 nsys 报告
  - `artifacts/4090d/` — RTX 4090 D (Ada, sm_89) 复测结果
  - `artifacts/5090/` — RTX 5090 (Blackwell, sm_120) 复测结果
    （`log_mxfp8_*/log_nvfp4_*/log_mxfp8_ts_*` 为软件模拟；
    `log_te114_*` 为 TE 1.14 per-tensor 对照；
    `log_te2_*` 为 TE 2.19 per-tensor / MXFP8 / NVFP4 recipe 对照）

详细设计与结果分析见 `REPORT.md`。
