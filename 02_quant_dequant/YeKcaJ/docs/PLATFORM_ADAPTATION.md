# 国产平台适配记录

## 当前状态

截至 2026-09-15：NVIDIA CUDA 保持通过回归；摩尔线程 MUSA 已完成真实 GPU 适配及归约/向量化、严格除法两轮优化，最新结果见 [摩尔优化记录](MUSA_OPTIMIZATION_LOG.md)。下方表格保留首次适配基线。MUPTI 采集被 MT-Perf 硬件连接问题阻断，完整设备内存检查工具缺失，不能声称这些检查已经完成。沐曦 MXMACA、华为昇腾尚未适配；适配验证不等同于厂商认证。

## 第 1 次适配：摩尔线程 MTT S4000

- 环境：MTT S4000，48 GiB，驱动和 MUSA SDK/mcc 5.1.0，架构 `mp_22`；设备查询用 `mthreads-gmi`。
- 源码基于 `mxfp8-YeKcaJ` 分支提交 `58099b85eb6ecf8012bbe4906216cdf9ba5500d1`，本次变更同步到远程 `/data/Learning-CUDA/02_quant_dequant/YeKcaJ`。
- 入口不变，运行和评估增加 `--backend musa`；使用 `Core/build-musa`，CUDA 默认使用 `Core/build`。
- MUSA kernel 位于 `Core/backends/musa/quantize.cuh`；公共编码、文件格式、CPU reference 和 frozen golden 保持原定义。

首次适配采用 shared memory 归约。后续查阅 SDK 并实测确认：设备报告 warpSize=128，但同步 shuffle 接口支持 32-lane 逻辑子组；现已迁移 shuffle 归约及 float4 加载，scale 广播仍保留一次 shared 同步。当前除法和乘法均用 FP64 中间结果一次 RN 转回 FP32，以保持数值规则；除法已替代最初的高开销 `__fdiv_rn`，百万对输入测试通过。

验证结果：

- 远程 MUSA、本地 CUDA 均 9/9 CTest 通过。
- 每种格式 299 个流程用例，含长度 0–65、255/256/257、1023/1024/1025、零、负零、极端有限值及冻结输入；编码器另测舍入边界和种子。
- 288 个长度/模式组合分别检查 data、scale、FP32/FP16/BF16 输出尾部哨兵；只能证明覆盖用例没有越界写，不能替代完整内存/竞争检查。
- 16,384 对 FP32 算术样本，32,768 次除法/乘法与 CPU 逐位一致（NaN 按类别比较）；归约独立测试通过。
- 144 组评估（3 种分布 × 2 种输入 × 2 种量化格式 × 2 种 scale × 2 种舍入 × 3 种输出）全部 CPU 量化/反量化对照通过；无非有限输出。
- 同时修复 CUDA MXFP8 向量写尾部不足 4 字节时的越界风险，不修改量化公式。

首轮性能（FP32 输入、block+nearest、预热 3 次、测量 20 次，单位 ms）：

| 格式 | 元素数 | 量化 median | 量化 P95 | 反量化 FP32 median | 反量化 FP16 median | 反量化 BF16 median |
|---|---:|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.681371 | 0.681989 | 0.091429 | 0.089817 | 0.090297 |
| MXFP8 | 4M | 2.560949 | 2.564411 | 0.148331 | 0.146903 | 0.146777 |
| MXFP8 | 16M | 10.092674 | 10.096823 | 0.473634 | 0.464857 | 0.464480 |
| NVFP4 | 1M | 0.617417 | 0.623863 | 0.100651 | 0.096000 | 0.095577 |
| NVFP4 | 4M | 2.092537 | 2.096754 | 0.226709 | 0.225771 | 0.226000 |
| NVFP4 | 16M | 8.093646 | 8.099040 | 0.738949 | 0.742126 | 0.743600 |

1M = 1,048,576 元素。以上是驻留 GPU 的 MUSA event 时间；NVFP4 量化包含全局两级归约，不含主机传输、分配和 CPU 校验。完整逻辑带宽、host_api、原始 JSONL、环境与源码哈希见 [实测记录](../records/musa-s4000-20260915/benchmark/RESULTS.md)。这些数字是适配基线，不是性能极限，也不能直接用 CUDA 的旧加速比描述。

构建/运行步骤及函数索引见 [MUSA 使用说明](../Core/backends/musa/README.md)。MUSA 专用 profiler、完整内存检查和进一步性能调优仍待开展。

## 适配目标

计划依次验证：

1. 摩尔线程 MUSA
2. 沐曦 MXMACA
3. 华为昇腾 CANN / Ascend C

## 保持不变的公共部分

- 输入文件和 v2 packed 文件格式；
- MXFP8/NVFP4 的 scale、编码、舍入公式；
- CPU reference/golden；
- FP32、FP16、BF16 输出定义；
- 误差、压缩率和日志字段。

## 每个平台需要适配的部分

- 设备内存申请、拷贝和释放；
- kernel 启动和 event 计时接口；
- `maximum`/`finalize_max` 归约；
- MXFP8/NVFP4 量化与 packed 写出；
- FP32/FP16/BF16 反量化输出；
- CMake/toolchain 和设备架构参数。

MUSA 和 MXMACA 可先评估 CUDA API 兼容层；昇腾需要使用 Ascend C 重写线程、片上内存和归约实现，不能只替换编译器名称。

## 验收标准

每个平台必须使用同一组 CPU golden 完成：

- packed data 逐字节一致；
- scale 逐字节一致；
- NVFP4 每字节两个 4-bit 元素，尾部高 nibble 为零；
- FP32/FP16/BF16 输出一致；
- 1M、4M、16M 性能数据；
- 最大绝对误差、MAE、MSE、压缩率和逻辑带宽；
- 平台、驱动、编译器和运行时版本记录。

## 性能优化口径

CUDA 版本的量化 kernel 已完成多轮优化。迁移到国产平台后不直接复用 CUDA 的性能结论：不同平台的带宽、向量宽度、线程模型和特殊函数吞吐不同，必须先完成正确性适配，再针对该平台重新 profile 和调优。workspace 复用、减少拷贝等端到端优化也需要在每个平台重新测量。

## 远程操作流程

```text
本地公共源码/CPU golden
    -> git push
远程平台 clone/pull
    -> 使用平台 SDK 编译
    -> 运行同一组 golden 和边界测试
    -> 保存性能/环境日志
    -> 回传结果并更新本文件
```

连接信息不写入仓库；使用 SSH key，不在脚本或文档中保存密码。
