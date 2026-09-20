# Hadamard 变换与量化融合

题目 3，训练营 ID：曹泽阳；提交目录：`zmuxuny`。实现 FP16/BF16 快速 Walsh-Hadamard 变换、MXFP8/NVFP4 融合量化，以及 FP16/BF16 Tensor Core 分解与融合路径（另保留 FP16 稠密 WMMA 对照）。本目录随附量化、文件读写和独立参考模块，可单独构建、测试和提交。

[总结报告](REPORT.md) 汇总功能、正确性、性能及平台结论；[工具分析](PROFILING.md) 给出 Profiler / Sanitizer 证据；[提交要求对应表](SUBMISSION.md) 列出交付内容。

## 使用

```bash
cd 03_hadamard_tc/zmuxuny
make
mkdir -p results/tmp
python3 tests/generate.py \
  --rows 8192 --cols 128 --dtype fp16 --output results/tmp/input.bin
./build/hadamard --input results/tmp/input.bin --batch 1 --seq 1024 --heads 8 \
  --output results/tmp/hadamard.bin --packed results/tmp/fused.bin \
  --unfused_packed results/tmp/unfused.bin --log results/tmp/metrics.json \
  --format nvfp4 --random_sign 1 --normalize 1 --repeats 100
python3 tests/validate.py
python3 tests/benchmark.py
```

依赖 CUDA Toolkit >=11.0、C++17 编译器及 Make，Python 脚本依赖见 `requirements.txt`。默认目标 `ARCH=75`；CUDA 11.5 优先使用 GCC 10，可通过 `HOSTCXX` 指定。切换架构前执行 `make clean`。文件格式见 [FILE_FORMAT.md](FILE_FORMAT.md)。输入张量文件使用二维行主序格式，逻辑形状为 `[batch, seq, heads, head_dim]`，前三维展平为 rows。`--batch * --seq * --heads` 必须等于 rows；默认是 `[1, rows, 1, cols]`。头维度支持 1–1024 的 2 的幂，包括验收所需的 64、128、256。

## 平台构建与实验

| 平台 | 构建 | 二进制目录 | 实测与分析 |
|---|---|---|---|
| NVIDIA RTX 3060 / 4090 D | `make ARCH=86` / `make ARCH=89` | `build/` | [4090 D 报告](REPORT_4090D.md) |
| MetaX C500 | `make PLATFORM=metax` | `build/metax/` | [C500 报告](REPORT_C500.md) |
| Iluvatar 智铠 100（MR-V100） | `make PLATFORM=iluvatar` | `build/iluvatar/` | [MR-V100 报告](REPORT_ILUVATAR.md) |
| Moore Threads MTT S4000 | `make PLATFORM=musa` | `build/musa/` | [S4000 报告](REPORT_MUSA.md) |
| Ascend 910B2 | `make PLATFORM=ascend` | `build/ascend/` | [昇腾报告](REPORT_ASCEND.md) |

天数实测 CoreX 4.4.0，默认 `COREX_PATH=/usr/local/corex`、`IVCORE_ARCH=ivcore11`；使用 CoreX clang 编译，运行前设置 `LD_LIBRARY_PATH=$COREX_PATH/lib64:${LD_LIBRARY_PATH:-}`。`make PLATFORM=iluvatar test` 包含数值验证和 49,152 项随机哈希一致性检查。复现性能、Profiler 和 Sanitizer 的命令见对应平台报告。

摩尔线程实测 MUSA 4.3.6 / `mp_22`，默认 `MUSA_PATH=/usr/local/musa`；运行前设置 `MUSA_VISIBLE_DEVICES=0` 和 `LD_LIBRARY_PATH=$MUSA_PATH/lib:${LD_LIBRARY_PATH:-}`。完整数值验证、参数扫描及工具采样命令见 [S4000 报告](REPORT_MUSA.md)。

昇腾实测 ARM64 / CANN 9.0 / 910B2，使用 Ascend C 独立内核和 ACL 运行时。先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，再运行 `make PLATFORM=ascend test`；测试、性能对照及 msprof / mssanitizer 说明见 [昇腾报告](REPORT_ASCEND.md)。

## 变换定义

采用自然顺序 Sylvester 矩阵 `H_1=[1]`，`H_2D=[[H_D,H_D],[H_D,-H_D]]`。默认输出 `y=x H_D / sqrt(D)`；`--normalize 0` 使用未归一化变换。`--random_sign 1` 在变换前沿最后一维应用固定 Rademacher 符号，所有行共用；`--sign_seed` 默认 7。

CUDA 及其兼容后端在 FP32 寄存器中计算，最终转换回输入 dtype。昇腾使用 UB 中的 FP32 标量/向量蝶形，矩阵分解路径直接使用 Mmad/Fixpipe；保留 H16 的 FP32 中间结果，再完成剩余蝶形和量化。每个逻辑线程组处理一行；NVIDIA 使用 32 线程组，MACA/CoreX 根据维度选择 32 或 64 线程组。前 log₂(组宽) 级通过 XOR shuffle 完成，其余级在同一线程的寄存器间完成。线程组数与矩阵 fragment 布局按平台调优，具体映射见对应实现文件的注释。

## 融合语义

`--format mxfp8` 使用 32 元素块，`--format nvfp4` 使用 16 元素块。CUDA 及兼容后端的融合路径在寄存器中完成 Hadamard、转换回 FP16/BF16、块最大值计算、缩放与打包；Ascend 使用 UB 中间张量及原生内核。**中间 dtype 舍入保留**，因此结果和先写出 Hadamard 结果、再调用本目录量化代码逐字节一致。支持 `--rounding nearest/stochastic` 和 `--seed`（默认 42）。

NVFP4 的全局 scale 依赖整个变换后的张量，不能仅凭单个 warp 的数据确定。实现先执行只计算变换后 amax 的 kernel，再执行融合输出 kernel。报告的 `fused_ms` 包含这次归约、清零和第二次变换；不会把先算 scale 的时间排除。MXFP8 不需要全局归约，CUDA 蝶形融合只需一次 kernel；矩阵路径的完整执行步骤随平台实现而异，全部计入对应日志耗时。

主程序同时运行非融合和融合路径，比较 scales、global scale、packed data；任何不一致都以非零状态退出。融合文件可由 `tests/reference.py` 的 `read_packed` / `dequantize` 读取，协议也与题目 2 兼容。

## Tensor Core 分解与融合

在支持的 GPU 上以 `make clean && make ARCH=89`（4090 D）或 `ARCH=86`（3060）构建即可开启，要求目标架构 `sm_80` 或更新。默认 `ARCH=75` 保留兼容实现。

分解路径将 `H_D` 分解为 `H_(D/16) ⊗ H_16`，使用两条 `mma.sync.m16n8k16` 处理 H16，其余级用 FP32 寄存器蝶形合并。FP16/BF16 均使用对应原生 MMA，累加为 FP32，复杂度 O(D log D)。进一步在寄存器内完成 MXFP8/NVFP4 量化；NVFP4 仍包含全局 amax 预遍历。

主程序默认测量蝶形和分解矩阵路径；CUDA 及兼容后端另有 FP16 稠密 WMMA 对照。昇腾的矩阵路径采用 H16 分解，支持 FP16/BF16。`--tensor_core 0` 可关闭 Tensor Core 测量。默认 `--output` 与 `--packed` 保存蝶形结果；指定下列参数可保存分解矩阵路径的结果：

```bash
./build/hadamard --input results/tmp/input.bin \
  --output results/tmp/butterfly.bin --packed results/tmp/butterfly.pack \
  --factorized_tc_output results/tmp/mma.bin \
  --factorized_tc_packed results/tmp/mma.pack \
  --log results/tmp/metrics.json --format nvfp4 --repeats 100
```

日志包含 `factorized_tc_ms`、`factorized_tc_unfused_ms`、`factorized_tc_fused_ms`、误差和加速比。每个后端独立比较其融合与非融合 packed 结果；Tensor Core 与蝶形因浮点求和顺序不同，分别与同一个 float64 参考比较。原稠密 WMMA 实现为 O(D²)，仅 FP16，仍以 `tensor_core_ms` 记录，`--tc_output` 保存其输出。两种 Tensor Core 算法在报告中明确区分。

## 测试与实验

- 独立 NumPy float64 稠密 Hadamard 矩阵验证，FP16 最大绝对误差 `<1e-2`、BF16 `<5e-2`。
- 覆盖所有支持头维度、两种 dtype、两种融合格式、非整 CTA 行数、零行、单位脉冲、归一化/未归一化及随机符号。
- 融合量化与非融合结果比较，并与 NumPy 独立量化参考比较。
- 基准覆盖小批量及 8192 行、多个头维度，保留三次独立运行数据，报告中位数。
- `rotation_quality.json` 在原坐标系比较直接量化与“随机旋转→量化→反变换”的 MSE，避免把不同坐标系的逐元素误差直接比较。

使用 CUDA event，预热 3 次；重复次数由 `--repeats` 指定。有效带宽按逻辑输入输出字节数计算。`unfused_ms` 包含变换和完整量化，`fused_ms` 包含融合所需的所有 GPU 步骤。小尺寸主要受启动开销影响，数据传输计时单独报告。实测分析见 [REPORT.md](REPORT.md)。

运行 `python3 tests/report.py` 可重建总收益数据及平台性能表；`tests/profile.py` 可独立运行 profiler/sanitizer 检查。`include/` 和 `tests/reference.py` 是本作者题目 2 数值模块的本地副本，两题的数值语义保持一致；保留副本是为使两份 PR 不依赖彼此的合并顺序。

原生 Linux 分析结果及本机 WSL 设置说明见 [PROFILING.md](PROFILING.md)。

## 软件优化与可选硬件转换

默认量化在整块对齐输入上使用每线程 4 元素向量读写，反量化按输出类型使用 4/8 元素；全局 amax 使用向量加载和 CTA 归约。FP8 软件最近偶数舍入通过整数进位完成。BF16 在 Ampere 及更新架构使用原生转换，较旧架构保留位运算实现。详情、消融与全部实测见 [4090 D 实验报告](REPORT_4090D.md)。

额外的 Ada FP8 转换对照使用独立可执行文件，不改变默认程序；需要 CUDA >=12.1 和 `sm_89` 或更新：

```bash
make native ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++
python3 tests/validate.py --binary build/hadamard_native
```

原生对照只加速 E4M3 最近偶数转换；随机舍入和 NVFP4 的 E2M1 编码继续使用软件实现。日志 `fp8_encoding` 标识所用实现。两路的 packed 文件、scales 和舍入语义一致。

`--materialized_compare 1` 可额外测量 NVFP4 的“变换并保留中间结果 + 量化”方案，日志字段为 `factorized_tc_materialized_ms`，可用 `--factorized_tc_materialized_packed PATH` 导出结果；默认仍为重算式融合。
