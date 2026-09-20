# 昇腾 910B2 适配、验证与性能报告

实测日期：2026-09-20。训练营 ID：曹泽阳。

完成原生 Ascend C 实现，92 组正确性测试通过；基础 memcheck 4 组和 msprof 4 组均完成。完整流程收益及最终方案见“总体优化结果”。

<!-- final-total-start -->
## 总体优化结果

128 MiB FP16 输入，默认块缩放、nearest。总加速比 = 基线耗时 / 最终耗时；耗时减少 = 1 − 最终耗时 / 基线耗时。

比较完整“变换＋量化”，两端分别选择已测最快方案，表内明确路径。方案需显式配置；此处不表示程序自动选择。

| 形状 / dtype | 格式 | 基线方案 → 最终方案 | 基线 ms | 最终 ms | 总加速比 | 耗时减少 |
|---|---|---|---|---|---|---|
| 65536×1024 / fp16 | mxfp8 | 蝶形融合 → 矩阵融合 | 337.40440 | 133.83687 | 2.52× | 60.3% |
| 65536×1024 / fp16 | nvfp4 | 蝶形非融合 → 矩阵非融合 | 569.70426 | 165.99768 | 3.43× | 70.9% |

<!-- final-total-end -->

## 环境与运行

单卡 Ascend 910B2，64GB HBM，ARM64 鲲鹏主机；CANN 9.0、毕昇编译器及驱动 26.1.1。ACL 使用容器内逻辑设备 0，`npu-smi` 显示物理卡 1。AIV 编译目标为 `dav-c220-vec`，Cube 为 `dav-c220-cube`；调度上限分别为 48 和 24 个 block。[完整环境](results/ascend/environment.txt)。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
make PLATFORM=ascend
make PLATFORM=ascend test
python3 tests/profile_ascend.py --tool profile
python3 tests/profile_ascend.py --tool sanitize
```

输入、配置、packed 文件和命令行与其他后端共用。`ascend/` 实现 Ascend C 内核及 ACL 运行时适配；主机侧保留原有文件校验和独立 C++ 参考。NPU 实测不依赖 PyTorch 或预装 MatMul 算子包。

## 实现与优化

- 蝶形路径在 UB 中执行 FP32 计算：步长 1/2/4 使用标量运算，步长 ≥8 使用向量 Add/Sub 和双缓冲，避免覆盖仍需读取的操作数。输入分块 DMA 搬入，每行只写回一次。
- 原生矩阵路径计算 `H_(D/16) ⊗ H16`。每个 Cube tile 处理 64×16 个输入；ND→NZ 搬运、LoadData、FP16/BF16 Mmad 和 Fixpipe 均由本项目调用。H16 权重在每个 core 中复用，尾 tile 按有效行数及 16 行对齐尺寸搬运，保持 FP32 累加。复杂度仍为 O(D log D)。
- Cube 的 FP32 中间结果经 GM 交给 AIV，后者完成剩余蝶形、归一化和最终 dtype 舍入。随机符号通过独立的位操作准备内核处理，其耗时包含在完整矩阵路径内。
- 融合保留中间 FP16/BF16 舍入。MXFP8 在 AIV 蝶形后直接编码；NVFP4 比较“归约后重算蝶形并编码”和“变换、amax、写回合并后再量化”。后者复用已写回的变换，避免重复蝶形。两种方案都包含全局归约的全部开销。
- 量化按整行批量写出 packed data 和 scale，减少每个小块的 DMA 启动；矩阵与向量阶段使用明确的流水线屏障，向量内核结束前恢复掩码。

Ascend 实现提供原生 H16 分解矩阵路径。其他后端的 O(D²) 稠密 WMMA 对照在本后端不启用；测试根据实际提供的路径验证输出，并强制要求 D≥16 的昇腾用例包含矩阵分解结果。

## 正确性与兼容性

完整独立参考测试 **92 组通过**，另有 1 组非法输入检查。[NPU 结果](results/ascend/correctness.json)、[RTX 3060 公共代码回归](results/ascend/correctness_nvidia_regression.json)。

两题的共享数值函数分别通过 **49,152 项哈希、65,536 项乘法、1,048,576 项除法**的主机/设备逐位对照，覆盖符号、指数边界、正规数、次正规数和舍入边界。[原始日志](results/ascend/numeric.log)。

其中 **76 组**覆盖 FP16/BF16 原生矩阵分解；最大绝对误差分别为 **0.0078125 / 0.015625**，低于 0.01 / 0.05。每个后端自己的融合/非融合，以及 NVFP4 写回方案均逐字节比较 packed data、scale 和 global scale。

## 同机性能对照

基线为[正确性通过的初始 Ascend C 移植](results/ascend/correctness_baseline.json)。基线/当前程序交替运行，每项 3 个独立进程试验取中位数；每条完整流程先预热 3 次，中小输入重复 10 次，128MiB 输入重复 3 次。使用 ACL 事件计时，包含全部 NPU kernel，排除分配、文件 I/O、主机参考与传输。计时保留启动提交间隔，不将其解释为单条硬件指令耗时。每次比较 packed 文件 SHA256。输入为 seed=42 的标准正态数据，最近偶数舍入，默认块缩放。独立变换使用归一化，性能对照关闭随机符号；开启随机符号的准备内核由正确性与工具采样另行覆盖。

[完整逐次记录及可执行文件/动态库 SHA256](results/ascend/comparison.json)。基线及全部可执行文件另存于完整实验备份；正式代码、构建日志和源码校验清单随提交提供。

| 形状 | dtype | 最终蝶形 ms | 最终矩阵 ms | 蝶形 / 矩阵 |
|---|---|---|---|---|
| 8192×1024 | fp16 | 15.05314 | 7.92818 | 1.90× |
| 65536×1024 | fp16 | 120.42426 | 63.55028 | 1.89× |
| 8192×1024 | bf16 | 16.69691 | 9.28187 | 1.80× |
| 65536×1024 | bf16 | 133.55643 | 74.34882 | 1.80× |

矩阵列包含 Cube、FP32 中间结果交接和 AIV 剩余蝶形，不能与仅计 Mmad 的时间比较。大尺寸完整量化流程如下：

| dtype / 格式 | 蝶形非融合 μs | 蝶形融合 μs | 矩阵非融合 μs | 矩阵融合 μs | 矩阵写回+amax μs |
|---|---:|---:|---:|---:|---:|
| fp16 / mxfp8 | 208545.53 | 191249.59 | 151694.92 | 133836.87 | — |
| fp16 / nvfp4 | 222864.95 | 308740.54 | 165997.68 | 191620.59 | 172297.44 |
| bf16 / mxfp8 | 234638.39 | 201802.49 | 175456.58 | 142044.28 | — |
| bf16 / nvfp4 | 251971.03 | 328916.77 | 192792.52 | 207141.03 | 198905.35 |

128MiB 的两种输入精度均以矩阵分解为优选：MXFP8 选择矩阵融合，NVFP4 选择矩阵非融合。写回并合并 amax 的方案减少一次启动，但该平台实验仍略慢于非融合；小尺寸需参考完整记录分别选择。FP32 中间结果的 GM 交接和软件编码仍有优化空间。

## Profiler 与 Sanitizer

msprof 采集 4 组格式/dtype 用例，导出任务时间与 PipeUtilization 计数器。采样与性能基准独立执行。[原始 CSV 与采样状态](results/ascend/profile/summary.json)。下表取 NVFP4/FP16 用例：33×1024，开启随机符号；同名 `ascend_had` 汇总多个 MODE 的调用，用于观察执行组成。

| kernel | 次数 | 平均 μs | 平均 AIV Scalar 比例 | 平均 AIC MAC 比例 |
|---|---:|---:|---:|---:|
| ascend_had | 50 | 89.965 | 98.6% | 0.0% |
| ascend_amax | 10 | 4.694 | 78.7% | 0.0% |
| ascend_max_finish | 25 | 4.667 | 98.7% | 0.0% |
| ascend_quant | 15 | 79.978 | 98.6% | 0.0% |
| ascend_prepare | 20 | 31.692 | 96.9% | 0.0% |
| ascend_cube | 20 | 17.725 | 0.0% | 0.4% |

量化内核的 AIV Scalar 比例为 98.6%；结合软件码本、缩放与逐元素舍入实现，当前优化重点仍在减少标量指令与 UB 标量访问。该比率属于所采样输入，不外推为所有尺寸的利用率。

Cube 的 MAC 计数器与原生 Mmad 实现共同核验矩阵路径已执行。小 tile 的搬运、流水线屏障，以及 Cube→GM→AIV 交接均有开销，因此报告比较整个变换和量化流程。

上述流水线比率来自工具，可能与其他流水线重叠，不能相加当作总时间分解。小规模采样用于核验执行路径；大尺寸结论使用独立基准。

mssanitizer 基础模式共 **4/4 次**满足：进程退出为零、实际 kernel 检查开始与完成次数一致、全部明确无错误且无警告。该平台实验实际执行基础 memcheck；racecheck/initcheck/synccheck 要求源码插桩，当前未能执行，不计作通过。[检查汇总及日志](results/ascend/sanitize/summary.json)。

完整 `--cce-enable-sanitizer` 源码插桩未通过本镜像的构建/运行验证，不能把基础模式结果表述为全量插桩通过。直接编译及最小样例触发毕昇后端 FrameIndex 错误；分阶段实验超时，原始记录见 [工具限制](results/ascend/tools_limitations/sanitizer_build_debug.log)。基础模式发现的向量掩码恢复警告已修复并复测。

## 误差实验与复核材料

常规实验使用每项 3 次试验、每次重复 5 次。[benchmark.json](results/ascend/benchmark.json) 保留误差、压缩率与逐次时间。矩阵实验覆盖 64–1024 维；[rotation_quality.json](results/ascend/rotation_quality.json) 在原坐标系比较直接量化和随机旋转后量化的重建 MSE。

[源码 SHA256](results/ascend/source_sha256.json)、[构建日志](results/ascend/build_final.log)、[使用说明](README.md)、[工具复现](PROFILING.md)。其他平台的归档数据对应各自原始版本；当前公共代码完整回归设备为 Ascend 910B2 与 RTX 3060。

API 依据：[Ascend C Fixpipe API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0251.html)、[msSanitizer 全量检查编译配置](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/compile_option_config.md)。实测编译使用服务器安装的 CANN 9.0 头文件。

## 基线复现

[还原补丁](results/ascend/baseline_restore.patch) 在本题目录的独立副本中使用：

```bash
baseline_dir=$(mktemp -d)
cp -r ascend include src tests Makefile "$baseline_dir/"
patch -d "$baseline_dir" -p1 < results/ascend/baseline_restore.patch
make -C "$baseline_dir" PLATFORM=ascend
python3 tests/benchmark_ascend.py --before "$baseline_dir/build/ascend/hadamard" --after build/ascend/hadamard --large
```

重建的 Hadamard 主程序、libcamp.so 和 libcube.so 均与实测基线 SHA256 相同。[重建哈希核对](results/ascend/baseline_rebuild.json)。
