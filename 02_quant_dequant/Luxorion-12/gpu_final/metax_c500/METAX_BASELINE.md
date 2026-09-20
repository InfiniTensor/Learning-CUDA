# C500 沐曦 baseline 记录

## 环境

- GPU：MetaX C500，当前实例为 50% sGPU，显存配额约 32 GiB
- 宿主运行时：MX-SMI 2.3.1，驱动 3.8.30，MACA 3.7.1.5
- 编译器：`mxcc`（由 `PATH` 或 `MXCC_BIN` 指定），版本 1.0.0
- 系统：Ubuntu 24.04.1，CMake 3.28.3，GCC 13.3.0，Ninja 1.11.1

## 已验证

- CPU reference：CMake 构建，6/6 测试通过。
- MXFP8 GPU oracle：4099 项检查通过。
- NVFP4 GPU oracle：12134 项检查通过。
- CLI 端到端：MXFP8 FP16、MXFP8 BF16 尾块、NVFP4 FP32、NVFP4 FP16 尾块均成功生成量化文件、恢复张量和 JSON 报告。
- C500 baseline 报告中的 `target_gpu` 已固定为 `MetaX C500`，不再沿用 NVIDIA 设备名。

## 构建方式

`scripts/build_metax.sh` 先构建 CPU reference，再通过 `mxcc` 编译 `.cu` 文件。不能使用 CMake 的 CUDA 语言模块，因为该模块会强制查找 `nvcc`；MXMACA 通过 `gpu_baseline/metax_include/` 提供 CUDA runtime/FP16/BF16 名称到 `mc*`/MACA 头文件的兼容层。

后续 v1–v8 优化必须在 C500 上重新测量；现有 RTX 3050 结果不作为性能结论。

## 方向 1 初测（C500）

同一 `normal_fp16_1024x1024_seed1234` 输入，预热 2 次、计时 20 次：

| 版本 | 量化 median | 反量化 median | 结论 |
|---|---:|---:|---|
| v0 MXMACA baseline | 0.478080 ms | 0.041856 ms | 当前最佳 |
| v1，128 threads/block | 0.686592 ms | 0.050560 ms | 未晋级，C500 上回退 |

v1 候选的输出误差统计与 baseline 一致，但性能明显回退，因此方向 2 应继续以 v0 为基础；不能直接继承 NVIDIA 上 v1 的晋级结论。

## 八方向首轮结果（同一 C500 输入）

以下均为 `normal_fp16_1024x1024_seed1234`，预热 2 次、计时 20 次的单轮筛选数据。所有候选输出误差统计与 v0 一致；是否晋级仍需在完整输入矩阵上复测。

| 版本 | 量化 median (ms) | 反量化 median (ms) | 首轮判断 |
|---|---:|---:|---|
| v0 baseline | 0.478080 | 0.041856 | 当前最佳综合基线 |
| v1 thread/block | 0.686592 | 0.050560 | 回退 |
| v2 coalescing | 0.686208 | 0.051072 | 回退 |
| v3 vectorized memory | 0.686080 | 0.041600 | 量化回退，不晋级 |
| v4 warp shuffle | 1.553152 | 0.044288 | 回退 |
| v5 shared reuse | 0.687360 | 0.042880 | 回退 |
| v6 bank conflict | 0.687104 | 0.043136 | 回退 |
| v7 branch reduction | 0.471680 | 0.042880 | 量化略快，反量化回退；暂不晋级 |
| v8 ILP/unroll | 0.479744 | 0.043008 | 回退 |

这组结果说明 NVIDIA 上的晋级链不能直接移植到 C500；首轮后仅 v7 值得进入完整复测。

## v0 / v7 完整复测与晋级

使用 12 个冻结输入（6 种分布 × FP16/FP32，包含非整齐尾块），3 轮交错运行；每次预热 5 次、计时 50 次。每个 v0/v7 配对的 `weights.luxq` 与恢复张量均逐字节一致，合计 72 个运行、0 个不一致。

| 指标 | v7 相对 v0 的几何平均 |
|---|---:|
| 量化 kernel | 1.01381× |
| 反量化 kernel | 1.00135× |

v7 的量化在全部 12 个输入上均更快，反量化的个别微小波动未形成综合回退。因此 v7（分支与编码搜索路径优化）晋级为当前 C500 最佳版本；v0 继续保留为基准。原始逐轮结果见 `optimization/metax_v0_v7_summary.csv`，三轮均值见 `optimization/metax_v0_v7_means.csv`；复现入口为 `scripts/benchmark_v0_v7_metax.sh`。

方向 1–8 的首轮候选还额外以相同 normal FP16 输入与 v0 逐字节比较 `weights.luxq` 和恢复张量：8/8 一致。因此其未晋级的原因均为性能，而非数值或文件格式错误。

## 性能统计口径修正

在提交性能表前修正并重新验证了以下统计问题，修正不改变 kernel 时间、量化字节或数值输出：

- `benchmark_main.cpp` 的压缩率分子改用输入文件实际 payload；FP16 的 `original_bytes` 不再按 FP32 的 4 bytes/element 计算。
- MXFP8 benchmark 分离量化和反量化的逻辑流量。反量化始终按 FP8 data + E8M0 scale 读取、FP32 输出写入统计，因此 FP16 输入不再低估反量化有效 GB/s。
- CLI 量化有效 GB/s 在 MXFP8 tensor scaling 时计入 global-amax 的额外一次输入读取；NVFP4 保持相同的两次读取口径。

在 C500 上重新构建并运行 `gpu_baseline_benchmark_metax input/benchmark build/metax/metric_validation 1 3` 后，所有 24 个（12 输入 × 2 格式）CPU/GPU 对照均通过。FP16 MXFP8 `normal` 的 `input_payload_bytes` 和 `original_bytes` 都为 2,097,152，量化 payload 为 1,081,344，压缩率为 1.939393939×。MXFP8 tensor FP16 CLI 的报告也按 `2*raw + payload` 计算量化逻辑流量。
