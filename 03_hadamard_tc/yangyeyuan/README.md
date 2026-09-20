# Fast Hadamard Transform + fused quantization (CUDA)

面向 `[batch, sequence, heads, head_dim]` 连续张量的归一化 Walsh–Hadamard
变换。支持 FP16/BF16 输入，`head_dim ∈ {64, 128, 256}`，以及逐行动态
FP8 E4M3、INT8 和 packed signed INT4 融合量化。

实现包含三条路径：

- `warp_fht`：每个 warp 处理一行，warp shuffle 完成前 5 级蝶形，寄存器完成
  余下蝶形，全程不使用共享内存；复杂度 `O(D log D)`。
- `shared_baseline`：每行一个 block、逐级共享内存同步，作为非 Tensor Core
  基线。
- `tensor_core_gemm`：通过 `cublasGemmEx` 计算 `X H_D / sqrt(D)`，作为
  Tensor Core 对照；复杂度 `O(D²)`。

融合量化核在寄存器中做完 FWHT 后继续执行 warp absmax 归约、scale 计算和
量化，不落地中间 FP16/BF16 张量。FP8 使用 E4M3 finite 范围 `[-448,448]`；
INT4 采用二补码 nibble：偶数维在低 4 位，
奇数维在高 4 位；量化范围 `[-7, 7]`，scale 为 `absmax / 7`。INT8 范围为
`[-127, 127]`。

## 构建

需要 NVIDIA GPU、CUDA Toolkit（含 cuBLAS）、CMake 3.24+ 和支持 C++17 的
编译器。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

把 `80` 换成目标 GPU 架构，例如 RTX 5060 Ti 使用 `120`。也可使用 Makefile：

```bash
make CUDA_HOME=/usr/local/cuda CUDA_ARCH=80 -j
```

若 CUDA 或 cuBLAS 不在标准路径，可分别设置 `CUDA_HOME` 和 `CUBLAS_ROOT`。

## 正确性测试

```bash
./build/hadamard_bench --self-test
# 或使用 CMake/CTest 构建时：
ctest --test-dir build --output-on-failure
```

自测覆盖 FP16/BF16、D=64/128/256、FP8/INT8/INT4 和非 8 倍数的行数。CPU FP32
参考采用相同 Sylvester 次序；FP16 最大绝对误差必须小于 `1e-2`，BF16 小于
`5e-2`。融合量化同时与“先 `warp_fht` 落地，再独立量化”的 GPU 结果逐字节
及逐 scale 比较。

## 基准

单个形状：

```bash
./build/hadamard_bench \
  --batch 4 --seq 128 --heads 32 --dim 128 \
  --dtype fp16 --quant int4 --warmup 50 --iters 500 \
  --csv results/performance.csv
```

完整矩阵：

```bash
./scripts/run_benchmarks.sh
```

计时只覆盖 GPU 工作，使用 CUDA events；分配、H2D/D2H、CPU 参考和 Tensor
Core Hadamard 矩阵构造均不计入。默认进行 `--rounds 5` 轮独立测量并取中位数，
抑制 boost clock 波动造成的离群值。`effective_GBps` 按最少的输入+输出字节计算，
不是 DRAM profiler 实测带宽。

可用 `--seed N` 融合一个确定性的随机 ±1 对角矩阵，计算 `H D_sign x`；seed=0
表示纯 Hadamard。密集 Tensor Core 对照目前只在 seed=0 时运行。

## Profiling

环境装有 Nsight Compute / Systems 时：

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

前者采集一个 `warp_fht_kernel` launch 的完整指标，后者采集 CUDA 与 cuBLAS
timeline。脚本输出到 `results/`，不会把 profiler 启动开销混入程序报告的
CUDA-event 时间。

注意：WSL2 的半虚拟化 GPU 不支持性能计数器与 kernel 级轨迹（ncu 报
`ERR_NVGPUCTRPERM`、nsys 无 kernel 数据），请在 Windows 宿主机或原生 Linux
上运行；原生 Linux 下如遇 `ERR_NVGPUCTRPERM` 则是权限问题，需 root 或调整
`NVreg_RestrictProfilingToAdminUsers`。受限环境下的实测记录与替代分析见
报告第 8 节。

详细设计、实测结果和后续方向见 [中文总结报告](docs/report.zh-CN.md)。

## 公共 CUDA API

声明位于 `include/hadamard.cuh`：

- `launch_fast` / `launch_baseline`
- `launch_fused_quant`
- `launch_quantize`（验证融合语义，也可独立使用）

所有 API 均接收 CUDA stream 并异步 launch；调用方拥有输入、输出和 scale
显存。无归一化版本没有隐式提供，本项目统一输出正交变换 `H/sqrt(D)`。
