# MXFP8 / NVFP4 低精度模拟与反量化

选题二实现：在普通 CUDA GPU 上完成低精度格式的编码、打包、缩放、解包、反量化与误差评估，不依赖 Hopper / Blackwell 的新硬件指令。同时支持 NVIDIA CUDA 与摩尔线程 MUSA 两个后端。

| 项目 | 支持范围 |
|---|---|
| 量化格式 | MXFP8（E4M3 元素 + E8M0 block scale）、NVFP4（E2M1 元素 + E4M3 block scale + FP32 global scale） |
| 输入类型 | FP32、FP16（FP16 在主机展开为 FP32 后上传） |
| 输出类型 | FP32、FP16、BF16 |
| 缩放策略 | `block`、`tensor` |
| 舍入方式 | `nearest`、`stochastic` |
| 后端 | CUDA（`Core/build`）、MUSA（`Core/build-musa`） |

正式程序只维护 `Core/`，全部操作统一从 `Core/tools/quantize.py` 进入。

总结报告：[项目总结报告.pdf](项目总结报告.pdf)。构建、测试及输入输出操作见下方步骤；历史优化轮次用于同配置趋势比较，最终冻结输入测试见 `records/controlled-v1/`。

---

## 目录作用

| 目录 / 文件 | 作用 | 说明文档 |
|---|---|---|
| `Core/` | 全部正式源码、配置、CPU 参考实现与测试 | [Core/README.md](Core/README.md) |
| `input/` | 输入张量，含固定性能输入与固定评估输入 | [input/README.md](input/README.md) |
| `output/` | **题目要求的三类交付输出**：低精度权重、反量化张量、误差日志 | [output/README.md](output/README.md) |
| `records/` | 每次实验的原始证据：benchmark JSONL、环境、nsys 报告、验证日志 | [records/README.md](records/README.md) |
| `docs/` | 题目原文 PDF、测试协议、平台适配记录 | [docs/README.md](docs/README.md) |
| `results/` | 本地保存的 nsys / ncu 报告与结论，体积大不入库 | [results/README.md](results/README.md) |
| `legacy/` | 结构调整前的旧实现存档，不参与构建 | [legacy/README.md](legacy/README.md) |
| `OPTIMIZATION_LOG.md` | 每轮 CUDA 优化的性能记录与加速比 | — |

`input/` 与 `output/` 只放数据和结果；源码一律在 `Core/`。

---

## 环境要求

Linux / WSL、CMake ≥ 3.18、CUDA Toolkit、C++17、Python ≥ 3.11。

`nvcc` 必须能被 CMake 找到。若出现 `No CMAKE_CUDA_COMPILER could be found`，说明 `nvcc` 不在 `PATH`，显式指定即可：

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$(which nvcc || echo /usr/local/cuda/bin/nvcc)
```

默认架构为 `sm_75`，自带 PTX，能在更高架构 GPU 上运行，但会触发 JIT，性能不具代表性。做性能测量时按实际显卡指定：

```bash
-DCMAKE_CUDA_ARCHITECTURES=89      # RTX 4090
-DCMAKE_CUDA_ARCHITECTURES=86      # RTX 3060
```

---

## 操作步骤

以下命令均在**项目根目录**执行。

### ① 构建并检查正确性

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
ctest --test-dir Core/build --output-on-failure
```

产物：`Core/build/pipeline_mxfp8`、`Core/build/pipeline_nvfp4`。

预期输出：`100% tests passed, 0 tests failed out of 9`（MUSA 后端为 10 项）。

### ② 生成输入

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 1024 --cols 1024
```

产物（终端会打印实际路径）：

```
input/<月日>/<编号>.fp32        输入张量，带头部，行主序
input/<月日>/<编号>.fp32.json   形状、分布、种子与 sha256
```

`--dtype` 可选 `fp32` / `fp16`；`--distribution` 可选 `uniform` / `normal` / `outlier`。

### ③ 量化并反量化

直接指定输入、量化格式、反量化精度和输出文件夹，无需修改配置文件：

```bash
python3 Core/tools/quantize.py run \
  --input input/917/1.fp32 \
  --format mxfp8 --output-type fp16 \
  --output-dir output/917 --save all
```

把示例输入换成 ②实际打印的路径。`--format` 可选 `mxfp8` / `nvfp4`；`--output-type` 可选 `fp32` / `fp16` / `bf16`。输入类型从文件头读取，支持 FP32/FP16。

上述命令直接在指定文件夹保存三类文件，终端打印绝对路径：

```
output/917/1_mxfp8_cuda_fp32_fp16.lpq     低精度权重：header + packed data + scale
output/917/1_mxfp8_cuda_fp32_fp16.fp16    反量化张量，行主序
output/917/1_mxfp8_cuda_fp32_fp16.json    误差、压缩率、kernel 时间、带宽
```

`--save all` 为默认值，保存题目要求的全部输出；也可用 `--save weights log` 仅保存权重和日志，或 `--save tensor` 仅保存反量化张量。选择只影响文件保存，完整计算、误差统计和 CPU 对照仍执行。题目交付使用 `all`。重复运行自动加 `_2`、`_3`，格式和后端也包含在文件名中。

不提供配置时默认 `cuda / fp32输出 / block / nearest / seed=1234`；可通过 `--backend`、`--output-type`、`--scale-mode`、`--rounding`、`--seed` 调整。block size 随量化格式固定为 32 或 16。

原配置文件入口仍可用，显式命令行参数优先于配置：

```bash
python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml \
  --input input/917/1.fp32 --output-type bf16
```

省略 `--output-dir` 和 `--prefix` 时沿用自动目录 `output/<月日>/<编号>/<格式>/<backend>/<输入dtype>_<输出dtype>.*`；其他位置的输入请指定 `--output-dir`。详见 [输出说明](output/README.md)。

### ④ 独立反量化已有权重

不需要原始输入或 CPU golden，直接从 `.lpq` 恢复：

```bash
./Core/build/pipeline_mxfp8 --dequant-file \
  output/916/1/mxfp8/cuda/fp32_fp32.lpq \
  /tmp/restored.bf16 bf16
```

最后一个参数可选 `fp32` / `fp16` / `bf16`；NVFP4 换成 `pipeline_nvfp4`。

评估默认自动分配输出目录；benchmark和profile显式指定的记录目录必须是新目录，避免覆盖旧记录。

### ⑤ 误差评估

```bash
python3 Core/tools/quantize.py evaluate --backend cuda
```

产物：

```
input/evaluation-v1/{uniform,normal,outlier}.{fp32,fp16}    六份固定输入
input/evaluation-v1/manifest.json                          输入参数与哈希
output/evaluation-v1/cuda/{uniform,normal,outlier}.json      每种分布48组误差
output/evaluation-v1/cuda/summary.json                      144组汇总
output/evaluation-v1/cuda/status.json                       完成状态与组数
output/evaluation-v1/cuda/<分布>/<格式>/*.{lpq,fp32,fp16,bf16,json}
```

覆盖3种分布×2种输入类型×2种格式×2种缩放×2种舍入×3种输出，共144组。每组保留权重、反量化张量和JSON日志，顶层只看三类误差汇总；文件名为 `<入>_<出>_<block或tensor>_<nearest或stochastic>`。

已有 `cuda/` 时自动使用 `cuda-2/`，不会覆盖旧结果。MUSA执行 `--backend musa`，共享上述输入，结果单独保存。`status.json` 中 `state=complete` 且 `completed=144` 才表示完成；中断也保留已完成组的汇总。可选 `--directory output/evaluation-v1/cuda-check` 指定新的结果目录，不要提前mkdir。

题目PDF第4页要求的三类文件与字段逐项对应见 [output/README.md](output/README.md)。输入参数和磁盘布局见 [input/README.md](input/README.md)。

### ⑥ 性能测试

```bash
python3 Core/tools/benchmark.py --directory records/my-benchmark-01 --repeats 20
```

产物：

```
records/my-benchmark-01/RESULTS.md              可读汇总
records/my-benchmark-01/summary.json            结构化汇总
records/my-benchmark-01/<格式>_<元素数>.jsonl   原始每次测量
records/my-benchmark-01/environment.json        GPU、驱动、编译器与源码哈希
```

正式优化必须 `--repeats 20` 并复用 `input/benchmark-v1/` 的冻结输入；试跑加 `--exploratory`。口径见 [固定协议](docs/BENCHMARK_PROTOCOL.md)。

### ⑦ 用 nsys 定位阶段

```bash
python3 Core/tools/profile.py --directory records/my-profile-01
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  records/my-profile-01/mxfp8.nsys-rep
```

产物（每种格式各一份）：

```
records/my-profile-01/mxfp8.nsys-rep      nsys 原始报告
records/my-profile-01/nvfp4.nsys-rep
records/my-profile-01/mxfp8.sqlite        nsys 导出的数据库
records/my-profile-01/nvfp4.sqlite
records/my-profile-01/mxfp8_capture.log   采集过程日志
records/my-profile-01/nvfp4_capture.log
records/my-profile-01/mxfp8_stats.txt     kernel 汇总统计
records/my-profile-01/nvfp4_stats.txt
records/my-profile-01/version.txt         nsys 版本
```

仅适用于 CUDA 后端。

### ⑧ 查看已有分析报告

kernel 内部指标（占用率、内存吞吐、停顿原因）见 [results/README.md](results/README.md)：

```bash
ncu --import results/ncu/mxfp8-test.ncu-rep --page details
```

---

## 其他平台

摩尔线程 MUSA 的构建、运行与诊断工具见 [Core/backends/musa/README.md](Core/backends/musa/README.md)。
适配进度与验收标准见 [docs/PLATFORM_ADAPTATION.md](docs/PLATFORM_ADAPTATION.md)。

## 性能结论

每轮优化的加速比、累计效果与测试条件见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。
性能测试的固定条件与加速比口径见 [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)。
