# Hadamard CUDA：A100跨卡验收报告

## 2026-09-08 A100 补充验证

现有融合源码已在 A100 sm80 重新编译，完整回归及三轮 52 配置验证通过。51/52 三轮更快，47/52 每轮至少减少 5%；一配置第三轮退化 8.43%，后续预热诊断没有重现，但不覆盖原负例。默认 original 保留。见[A100 报告](fused-layout-a100-validation.md)和[原始数据](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_fused_a100_20260908/README.md)。以下历史实验和适用范围分别保留。

**固定源码已在真实A100重新编译并完成全部预定验证。**源码为`12c76d8331ef7cf3fd4c8c14a049162559be4302`，参考库为`e7706faf8d1c3b9f241e36860640ad1dac644ede`。完整公开证据、源码和离线复算入口见[档案首页](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/README.md)。

## 环境与构建

实际设备为NVIDIA A100-SXM4-40GB、MIG Disabled、capability `[8,0]`，驱动550.127.05。独立环境复用Conda Python3.12.7、Torch2.5.0+cu124、NumPy2.1.3，使用NVCC12.4.99、G++11.4与C++11 ABI=false单并发构建；没有升级原Torch或修改其他项目环境。

CLI、默认扩展、线程扩展的cuobjdump均显示`sm_80.cubin`；固定Dao含sm75/80/87/90。来源和构建参数见[源码清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/source_manifest.json)及[实际ELF检查](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/retrieved/results/collection_metadata.json)，没有将sm89产物当作A100验收。

| 二进制 | 实际SHA256 |
|---|---|
| CLI | `aadc59aa1e8c2ba40dd0763c89db945b0dc53f5f35699828162f1381b13c6263` |
| 默认扩展 | `4f1f44550a6e57b39ec654394f6175048bb806f6d08d1e3989473c674ce5ddb0` |
| 线程扩展 | `0857075b7199d2040457da62610a98daec4f6cd6d4ae1cb0a3972cc16737291d` |
| Dao参考 | `faa31572adfbf82c0261a27ebfaed6a2c5fb6261b81a499e9d3a9bc4819726e5` |

## 正确性：不同范围分别记账

| 验证范围 | A100结果 |
|---|---|
| 原1,876组CLI矩阵 | 默认128、显式256均PASS；最大绝对误差0.0078125；CPU/分步/融合INT4 bytes与scales精确一致 |
| 原1,800组Dao矩阵 | PASS；FP16/BF16最大绝对差均0；既有10项非法张量和非默认stream检查通过 |
| 同一1,800组调用兼容验证 | 旧默认、显式128、显式256位一致；27项无效线程值拒绝通过 |
| 候选范围336个不同输入 | 三轮均通过；覆盖CPU全量量化、样本行FP64稠密参考和捕获输出核查 |

CLI保留35个warp相对稠密舍入参考的容差内差异，没有修改期望或放宽阈值。重复1,876矩阵、三种API调用和三个进程均不计作更多独立算法用例。当前仅一张可见GPU，多GPU切换未实际覆盖。

## 默认128线程：原12组对照及负例

原12组eager API的`Dao时间/本项目时间`为2.295～2.380；事件区间含主机调用造成的GPU空档，**不能称纯kernel加速比**。

相同输入的64份独立输出CUDA Graph，比值为0.9895～2.1297；两个负例完整保留：

| 形状与dtype | 本项目默认128（ms） | Dao（ms） | 本项目耗时增加 |
|---|---:|---:|---:|
| `[17,256]` FP16 | 0.0017600 | 0.0017416 | 1.06% |
| `[17,256]` BF16 | 0.0017592 | 0.0017488 | 0.59% |

[所有12组eager/12组Graph](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/derived/dao_all_configurations.csv)包含各组样本和中位数；小差异不是跨日期统计结论。CLI另保存16组配置、110行kernel-only/CPU compute/host end-to-end数据，不能与API/Graph直接相除。

## 显式256线程：预定72配置三轮

原24个目标和48个M±1邻近配置，均在每轮相对同轮128线程减少耗时至少5%。**72/72达标，无观测退化**；全范围降时6.19%～26.09%。

| 原目标范围（两精度、两scale、三轮） | 耗时减少范围 |
|---|---:|
| transform，M=4096，N=16 | 13.68%～14.61% |
| transform，M=4096，N=64 | 11.07%～12.25% |
| transform，M=16384，N=16 | 25.75%～26.09% |
| transform，M=16384，N=64 | 15.17%～16.52% |
| fused INT4，M=4096，N=16 | 8.21%～8.68% |
| fused INT4，M=4096，N=64 | 6.28%～7.02% |

邻近范围仅为上述M±1；融合未扩大到M=16384。图中每方法保留64份独立输出，25次API预热、5次图预热、每组20次重放、5组，三进程轮换测量顺序。[全部72配置](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/nvidia_a100_20260906/derived/promotion_all_configurations.csv)及三份原始JSON均保留。

默认仍为128，没有自动派发。正确性支持全部已实现N，但此次性能结论只支持预定N=16/64范围，不声称全局最优或其他形状必然更快。

## 跨卡比较与结论边界

A100与4090若作横向观察，只能比较一致的测量口径，例如相同64独立输出的Graph设置，或相同CLI的kernel-only；不能将A100的Graph与4090的eager、CPU或含复制时间相除。两端Torch、CUDA、CPU和时钟条件并未全部控制，不能把绝对差异仅归因于GPU型号。

本报告的所有加速/降时比值均取自**同轮A100内部配对**，没有用4090数值作分母。Graph时间包含均摊调度开销，既不是独立单kernel延迟，也不是端到端速度；logical tensor I/O不等同物理内存带宽。

运行于2026-09-06北京时间02:55:05～03:05:35。最终GPUUtil=0%、显存14MiB、compute-app列表为空，全部阶段退出0。首次归档时，48 份原始文本的大小和 SHA 均核验且保留原字节；当前公开档案保留其中 47 份原字节，另 1 份实验脚本仅整理模块说明文字。历史执行 hash 不变，文字编辑副本和派生路径处理分别记录原/公开双 SHA，见 [文字来源说明](public-wording-provenance.json) 与归档内的 public_derivation_provenance.json。没有公开访问地址、连接材料或编译二进制。

本报告用于核对 A100 的数值、性能与复现记录。后续按已测慢例实施优化；PR 和课程登记在项目所有者验收具体代码及证据后单独执行。
