# Iluvatar 智铠 100（MR-V100）适配与性能报告

实测日期：2026-09-20。训练营 ID：曹泽阳；提交目录：`zmuxuny`。

<!-- final-total-start -->
## 总体优化结果

128 MiB FP16 输入，默认块缩放、nearest。总加速比 = 基线耗时 / 最终耗时；耗时减少 = 1 − 最终耗时 / 基线耗时。

比较完整“变换＋量化”，两端分别选择已测最快方案，表内明确路径。方案需显式配置；此处不表示程序自动选择。

| 形状 / dtype | 格式 | 基线方案 → 最终方案 | 基线 ms | 最终 ms | 总加速比 | 耗时减少 |
|---|---|---|---|---|---|---|
| 65536×1024 / fp16 | mxfp8 | 矩阵非融合 → 矩阵非融合 | 0.99989 | 0.99969 | 1.00× | 0.0% |
| 65536×1024 / fp16 | nvfp4 | 矩阵非融合 → 矩阵写回+amax | 1.57946 | 1.10027 | 1.44× | 30.3% |

<!-- final-total-end -->

## 环境与复现

单卡 Iluvatar MR-V100，32GiB 显存，16 个计算单元，原生 warp 为 64 线程。
Ubuntu 24.04.4、CoreX SDK / 驱动 4.4.0、Python 3.12、NumPy 1.26.4；CPU 配额 12 核，内存 32GiB。使用 CoreX clang 的 `-x ivcore --cuda-gpu-arch=ivcore11 -O3 -ffp-contract=off`。
[环境原始记录](results/iluvatar/environment.txt) · [设备属性](results/iluvatar/device.json)。

```bash
export COREX_PATH=/usr/local/corex
export LD_LIBRARY_PATH="$COREX_PATH/lib64:${LD_LIBRARY_PATH:-}"
make PLATFORM=iluvatar
make PLATFORM=iluvatar numeric-test
python3 tests/validate.py --binary build/iluvatar/hadamard --output results/iluvatar/correctness.json
python3 tests/benchmark.py --binary build/iluvatar/hadamard --output-dir results/iluvatar --trials 3 --repeats 100 --materialized-compare
python3 tests/profile_iluvatar.py
python3 tests/report_iluvatar.py
python3 tests/plot_iluvatar.py
```

默认 `PLATFORM=nvidia`；天数二进制单独存放在 `build/iluvatar/`。软件 FP8/FP4 编码及文件协议与其他后端相同，NVIDIA 专用 PTX 不进入 CoreX 编译路径。

## 设计

- 量化保留 32/16 元素逻辑缩放组；FP32/MXFP8 使用每线程 2 元素，较大 FP16/NVFP4 使用 8 元素。小张量独立选择启动配置，65536 元素分派边界有测试覆盖。
- 全局 amax 对 FP32 使用标量连续访问；16 位输入使用 4 元素向量及受限网格。计时包含清零、全局归约及最终量化。
- BF16 转换、负零符号和整数钳位同时支持 host/device。CoreX 4.4 在优化 lane 索引哈希时可错误复用移位结果；仅该哈希的设备局部变量使用 `volatile`，并以 49,152 次逐项测试覆盖线程边界、种子和超过 2³² 的索引。未改变随机序列或舍入语义。
- 原生 FP16/BF16 矩阵指令实现 H16，其余因子以 FP32 蝶形完成，复杂度 O(D log D)。实现位于 [iluvatar_matrix.cuh](src/iluvatar_matrix.cuh)，并非用蝶形冒充矩阵路径。
- 令 `g=lane/16, c=lane%16`。输入 A 每线程元素位于 `(2g+j%2+8⌊j/2⌋, c)`，B 使用相同的 K 索引；FP32 累加器位于 `(g+4j, c)`。映射依据 CoreX 4.4 的 `crt/iluvatar_mma.hpp`；与 NVIDIA/MACA 的 fragment 分开实现，避免直接复用不兼容布局。
- 蝶形按阶段选择 32/64 线程组，amax 预遍历使用更多行/CTA，D≥512 的输出阶段使用两个 64 线程组以降低寄存器压力。稠密矩阵对照使用补偿求和，避免长串 FP32 加法在未归一化 FP16 舍入中点处产生大于阈值的误差。
- 融合前保留中间 FP16/BF16 舍入，packed 结果与各自非融合流程逐字节一致。另提供 NVFP4 的变换+amax 写回、再量化方案。默认输出仍为蝶形；矩阵输出使用 `--factorized_tc_output` / `--factorized_tc_packed`，两步量化使用 `--materialized_compare 1 --factorized_tc_materialized_packed <文件>`。
- 矩阵与蝶形启动扫描：[tune_mma.csv](results/iluvatar/tune_mma.csv)、[tune_butterfly.csv](results/iluvatar/tune_butterfly.csv)。

## 验证

独立 NumPy 参考：**92 组全部通过**，见 [correctness.json](results/iluvatar/correctness.json)。
其中 76 组覆盖分解矩阵路径；FP16/BF16 最大绝对误差为 0.0078125 / 0.015625，分别低于 0.01 / 0.05。覆盖尾行、1–1024 维、归一化/随机符号、随机舍入与 NVFP4 两步方案；另验证非法尺寸拒绝。
RTX 3060 / sm_86 完整回归 **92 组通过**：[记录](results/iluvatar/correctness_nvidia_regression.json)。设备/主机随机哈希的精确比较见 [numeric.log](results/iluvatar/numeric.log)。
设备检查：**12 次 ixsan 全部通过**，涵盖 memcheck、racecheck、initcheck；逐次命令、退出码、零错误汇总和二进制 SHA256 见 [检查汇总](results/iluvatar/profile/summary_sanitizer.json)。该工具未提供 synccheck。

## 性能对照

在同一张 MR-V100 上交替执行基线与当前版本，每项 3 次独立试验，取设备事件计时的中位数。预热 3 次，小规模重复 100 次，大规模重复 30 次。包含完整 GPU 流程，排除文件读写、主机传输和 CPU 参考。每次验证 packed SHA256 一致。
基线：MR-V100 correct initial native matrix port: 32-lane butterfly, four matrix waves per CTA, original quantization geometry。完整逐次时间与二进制哈希见 [comparison.json](results/iluvatar/comparison.json)；[initial_port.patch](results/iluvatar/initial_port.patch) 可在本目录副本中通过 `patch -p1 < results/iluvatar/initial_port.patch` 还原正确性已通过的初始移植源码，再执行 `make -B PLATFORM=iluvatar`。
基线和当前程序可用 `tests/benchmark_tuning.py --before <基线> --after <当前> --output <JSON> --trials 3` 重新比较。补丁包含该基线所需的构建和平台兼容代码。

### 独立变换

| 形状 | dtype | 最终蝶形 ms | 最终矩阵 ms | 蝶形 / 矩阵 |
|---|---|---|---|---|
| 8192×64 | fp16 | 0.01067 | 0.00821 | 1.30× |
| 8192×128 | fp16 | 0.01292 | 0.01197 | 1.08× |
| 8192×256 | fp16 | 0.02044 | 0.01978 | 1.03× |
| 8192×512 | fp16 | 0.03605 | 0.03242 | 1.11× |
| 8192×1024 | fp16 | 0.12311 | 0.06646 | 1.85× |
| 8192×64 | bf16 | 0.01012 | 0.00715 | 1.42× |
| 8192×128 | bf16 | 0.01263 | 0.01131 | 1.12× |
| 8192×256 | bf16 | 0.02097 | 0.01964 | 1.07× |
| 8192×512 | bf16 | 0.03775 | 0.03324 | 1.14× |
| 8192×1024 | bf16 | 0.12768 | 0.06884 | 1.85× |
| 65536×1024 | fp16 | 0.89576 | 0.48758 | 1.84× |
| 65536×1024 | bf16 | 0.92460 | 0.48841 | 1.89× |

### 矩阵变换与量化

| 形状 | dtype | 格式 | 非融合 μs | 全融合 μs | 变换+amax 写回后量化 μs |
|---|---|---|---:|---:|---:|
| 8192×128 | fp16 | mxfp8 | 27.48 | 25.26 | — |
| 8192×128 | fp16 | nvfp4 | 37.89 | 55.76 | 39.31 |
| 8192×1024 | fp16 | mxfp8 | 133.04 | 143.96 | — |
| 8192×1024 | fp16 | nvfp4 | 188.26 | 274.75 | 156.33 |
| 8192×128 | bf16 | mxfp8 | 24.51 | 23.23 | — |
| 8192×128 | bf16 | nvfp4 | 33.15 | 53.19 | 35.85 |
| 8192×1024 | bf16 | mxfp8 | 138.03 | 146.22 | — |
| 8192×1024 | bf16 | nvfp4 | 190.83 | 280.17 | 159.49 |
| 65536×1024 | fp16 | mxfp8 | 999.69 | 1090.98 | — |
| 65536×1024 | fp16 | nvfp4 | 1311.26 | 2053.67 | 1100.27 |
| 65536×1024 | bf16 | mxfp8 | 994.97 | 1091.94 | — |
| 65536×1024 | bf16 | nvfp4 | 1301.88 | 2092.62 | 1117.08 |

NVFP4 全融合需要先求全局 amax，再重新变换；两步方案写回中间结果，避免第二次变换。最优路径按尺寸和格式比较，不能由 kernel 数量直接判断。独立变换表比较最终实现的矩阵与蝶形；完整流程的总收益见上表。
正常分布和离群值分布的旋转重建误差见 [rotation_quality.json](results/iluvatar/rotation_quality.json)。

## Profiler

ixsys 原生时间线与 ixkn 硬件计数器均已采集。以下为插桩统计，不能与上面未插桩基准直接比较：[时间线](results/iluvatar/profile/summary_trace.json)、[计数器](results/iluvatar/profile/summary_counters.json)、[机器可读分析](results/iluvatar/profile/analysis.json)。

| 用例 | kernel | 次数 | 平均 μs（插桩） |
|---|---|---:|---:|
| trace_mxfp8_fp16 | `hadamard_mma_kernel<1024, 1, 0, 0, 4>` | 12 | 66.056 |
| trace_mxfp8_fp16 | `hadamard_mma_kernel<1024, 1, 2, 0, 1>` | 6 | 147.429 |
| trace_nvfp4_fp16 | `hadamard_mma_kernel<1024, 1, 0, 0, 4>` | 12 | 65.807 |
| trace_nvfp4_fp16 | `hadamard_mma_kernel<1024, 1, 1, 1, 4>` | 6 | 60.965 |
| trace_nvfp4_fp16 | `hadamard_mma_kernel<1024, 1, 2, 1, 4>` | 6 | 214.393 |
| trace_nvfp4_fp16 | `hadamard_mma_kernel<1024, 1, 3, 1, 4>` | 6 | 72.468 |
| trace_mxfp8_bf16 | `hadamard_mma_kernel<1024, 2, 0, 0, 4>` | 12 | 67.962 |
| trace_mxfp8_bf16 | `hadamard_mma_kernel<1024, 2, 2, 0, 1>` | 6 | 147.685 |
| trace_nvfp4_bf16 | `hadamard_mma_kernel<1024, 2, 0, 0, 4>` | 12 | 68.723 |
| trace_nvfp4_bf16 | `hadamard_mma_kernel<1024, 2, 1, 1, 4>` | 6 | 63.964 |
| trace_nvfp4_bf16 | `hadamard_mma_kernel<1024, 2, 2, 1, 4>` | 6 | 217.124 |
| trace_nvfp4_bf16 | `hadamard_mma_kernel<1024, 2, 3, 1, 4>` | 6 | 75.846 |

| 计数器用例 / kernel | TCU 效率 | 实测占用率 | 矩阵指令数 |
|---|---:|---:|---:|
| counters_mxfp8_fp16 / matrix_transform | 3.6464 % | 63.38 % | 32768 Instructions |
| counters_mxfp8_fp16 / matrix_fused | 1.9589 % | 59.51 % | 32768 Instructions |
| counters_nvfp4_fp16 / matrix_transform | 3.6497 % | 63.42 % | 32768 Instructions |
| counters_nvfp4_fp16 / matrix_fused | 1.3008 % | 59.33 % | 32768 Instructions |
| counters_nvfp4_fp16 / matrix_amax | 4.9885 % | 57.39 % | 32768 Instructions |
| counters_nvfp4_fp16 / matrix_materialized | 4.0140 % | 64.90 % | 32768 Instructions |
| counters_mxfp8_bf16 / matrix_transform | 3.5392 % | 63.81 % | 32768 Instructions |
| counters_mxfp8_bf16 / matrix_fused | 1.8706 % | 58.55 % | 32768 Instructions |
| counters_nvfp4_bf16 / matrix_transform | 3.5537 % | 63.73 % | 32768 Instructions |
| counters_nvfp4_bf16 / matrix_fused | 1.2833 % | 59.51 % | 32768 Instructions |
| counters_nvfp4_bf16 / matrix_amax | 4.7582 % | 57.21 % | 32768 Instructions |
| counters_nvfp4_bf16 / matrix_materialized | 3.8083 % | 64.77 % | 32768 Instructions |

计数器结果来自单次 kernel replay；指令计数用于核实矩阵路径，占用率用于观察当前启动配置，不单独作为速度判断依据。

大维度样本为 8192×1024：独立变换与每次融合变换均记录到 32,768 条矩阵指令，与 256 元素/H16 tile 的分解一致。NVFP4 全融合的 amax 预遍历和最终写出各执行一次变换；写回方案在一次变换中同时得到输出和 amax。软件编码、寄存器中间值及写出方式也不同，因此减少 kernel 数量并不保证降低总时间。以上完整流程计时支持大尺寸选择非融合 MXFP8 / 两步 NVFP4 的结论。
