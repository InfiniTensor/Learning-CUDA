训练营 ID：**曹泽阳**；GitHub ID：`zmuxuny`；提交目录：`03_hadamard_tc/zmuxuny/`。

实现 FP16/BF16 Hadamard 变换及 MXFP8/NVFP4 量化融合，支持头维度 1–1024 的 2 的幂、归一化及固定随机符号。提供蝶形与原生矩阵指令路径，并比较融合和分步执行的完整成本。本目录随附量化与文件模块，可独立构建和测试。

### 实现

蝶形按 `(a,b) → (a+b,a-b)` 分 log₂D 级处理，使用 FP32 中间量。分解矩阵路径通过原生 FP16/BF16 指令计算 H16，再由蝶形完成其余因子，维持 O(D log D)；另保留 FP16 稠密 WMMA 对照。

融合量化保留中间 FP16/BF16 舍入，与自身非融合路径逐字节一致。MXFP8 可单遍融合；NVFP4 比较“重算变换求全局 amax”“写回时同时求 amax”“独立变换再量化”三种流程，计时均包含全部必要步骤。

提供 NVIDIA CUDA、MACA（C500）、CoreX（MR-V100）、MUSA（S4000）及 Ascend C（910B2）后端。各平台独立处理矩阵 fragment 与线程布局；Ascend 使用原生 Cube Mmad/Fixpipe 和 AIV 后处理。默认文件输出为蝶形结果，矩阵路径用独立参数导出。

### 正确性

当前计算实现的 Ascend 910B2 与 RTX 3060 回归各通过 **92 组测试 + 1 组非法尺寸检查**，其中 76 组覆盖分解矩阵。独立 float64 Sylvester 参考测得 FP16 / BF16 最大绝对误差 **0.0078125 / 0.015625**，分别低于题目 0.01 / 0.05；融合的 packed data、scales、global scale 与同算法非融合路径一致。

### 总体性能收益

128 MiB FP16/NVFP4，完整变换＋量化，单位 ms。总加速比 = 起始基线耗时 / 最终耗时；耗时减少 = 1 − 最终耗时 / 基线耗时。

| 平台 | 基线方案 → 最终方案 | 基线 ms | 最终 ms | 总加速比 | 耗时减少 |
|---|---|---|---|---|---|
| RTX 4090 D | 蝶形融合 → 矩阵融合 | 1.09036 | 0.35407 | 3.08× | 67.5% |
| C500 | 矩阵非融合 → 矩阵写回+amax | 1.14481 | 0.83216 | 1.38× | 27.3% |
| MR-V100 | 矩阵非融合 → 矩阵写回+amax | 1.57946 | 1.10027 | 1.44× | 30.3% |
| S4000 | 蝶形非融合 → 蝶形非融合 | 37.06536 | 3.97626 | 9.32× | 89.3% |
| Ascend 910B2 | 蝶形非融合 → 矩阵非融合 | 569.70426 | 165.99768 | 3.43× | 70.9% |

NVIDIA 起始基线为 `0a5e40b`；国产平台为同设备正确性通过的基础移植。NVIDIA 为同卡同配置分批测量，国产平台为同机交替测量，均取 3 次独立试验中位数，设备事件计时排除主机传输。两端各取已测最快完整方案，路径如表所列；矩阵/蝶形及融合/非融合的最终方案对照在报告中单列。两种格式、更多尺寸及原始数据来源见 [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT.md)。

### 平台证据与复现

| 内容 | 材料 |
|---|---|
| 构建、输入形状、各路径运行和输出参数 | [README](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/README.md) |
| 算法、误差、矩阵/蝶形、融合及 CPU 对照 | [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT.md) |
| 各平台实验 | [NVIDIA](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_4090D.md)、[C500](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_C500.md)、[MR-V100](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ILUVATAR.md)、[S4000](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_MUSA.md)、[Ascend](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ASCEND.md) |
| 指令证据、Profiler 与 Sanitizer 范围 | [工具分析](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/PROFILING.md) |

NVIDIA 有 Nsight Systems 与矩阵指令反汇编，软件及原生两路共 36 次 Sanitizer 检查通过；MR-V100 有矩阵硬件计数器及 12 次检查；Ascend 有 msprof 与 4 次基础 memcheck。其他平台工具可用范围和失败记录在工具分析中列明。

各平台性能与工具结果对应报告中的实验版本。当前计算实现完整回归设备为 Ascend 910B2 和 RTX 3060，其余设备保留原版本证据。原始 JSON、逐次计时、源码及二进制指纹随提交提供；本 PR 仅包含题目 3 目录，不依赖题目 2。
