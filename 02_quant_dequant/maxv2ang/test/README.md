# 测试

在有 GPU 的机器上，工程根目录一行命令跑完全套：

```bash
make test          # 全流程（含 nsys / ncu，工具可用时自动跑）
make test-fast     # 跳过 nsys / ncu，只出正确性与性能数据（快）
```

`-arch` 由 Makefile 按 GPU 计算能力自动探测，不需要手动指定；需要强制时
`make test ARCH="-arch=sm_86"`。

## 目录分工

| 位置 | 角色 |
|---|---|
| `../include/` + `../src/` | **库**：格式逻辑与 GPU 实现。不含 main、不含命令行、不产出可执行文件 |
| `test/test_all.cpp` | **测试主程序**：本工程唯一编译成可执行文件的地方 |
| `test/anchors.h` | 格式锚点自测（次正规数、特殊值码点、舍入平局、打包字节序） |
| `test/gen_input.h` | 三种分布测试矩阵的生成 |
| `test/run.sh` | 只做一件事：调用 nsys / ncu 两个外部分析工具 |

测试程序调用的是库里的公开接口（`gpu_init` / `quantize_gpu` / `dequant_gpu` /
`run_one`，见 `include/quant_gpu.h`），所以测的就是交付的那份代码，
不存在"测试另写一套"。

库自身的分层：

```
include/quant_format.h   格式语义（__host__ __device__，header-only）
                         ↑ 设备编译唯一需要看到的头文件
include/quant_tables.h   常量表（主机副本 + __constant__ 设备副本）
include/quant_io.h       主机侧 API（配置 / 文件 I/O / 主机参考实现 / 指标）→ src/quant_io.cu
include/quant_gpu.h      GPU 库 API → src/quant_gpu.cu
```

## 流程（`test/test_all.cpp`）

| 步骤 | 内容 | 失败处理 |
|---|---|---|
| 0 | 环境：`gpu_init()` + GPU 名称 / 算力 / SM 数 | 中止 |
| 1 | **正确性自测**：格式锚点（29 项）+ GPU 全流程逐字节对拍（6 种格式组合 × 3 组数据） | **中止（先对，再快）** |
| 2 | 生成输入：性能档 1024×1024 fp32 ×3 分布；覆盖档 256×256 ×{fp32,fp16} ×3 分布 | 中止 |
| 3 | **性能档**：1024×1024 fp32→fp16，两格式（走 `read_config`，用示例配置文件） | 中止 |
| 4 | **覆盖档**：题目要求的输入/输出维度全遍历（见下表） | 中止 |
| 5 | 边界尺寸 257×131（非整块，覆盖尾部块路径） | 中止 |
| 6 | 汇总：性能表 + 覆盖表 + 逐组合完整指标，落盘 `summary.txt` | — |

第 3–5 步都走库里的 `run_one(..., verify=true)`：GPU 量化字节与反量化输出都要与
主机端实现逐字节一致，不一致则 `run_one` 返回 -1，程序立即中止。

### 覆盖档：题目要求的维度逐格跑通

| 维度 | 取值 | 覆盖 |
|---|---|---|
| 输入 dtype | fp32 / fp16 | ✅ 题目任务 1「读取 FP32/FP16 输入矩阵」 |
| 输出类型 | fp16 / bf16 / fp32 | ✅ 题目任务 4「反量化为 FP16/BF16/FP32」 |
| 舍入模式 | nearest / stochastic | ✅ stochastic 走 GPU 全流程，验证确定性 RNG 下设备端与主机端仍逐字节一致 |
| 缩放模式 | block / tensor | ✅ 覆盖档用 block；tensor 由第 1 步的 GPU 自测覆盖 |
| 元素格式 | E4M3 / E5M2（MXFP8） | ✅ 覆盖档用 E4M3；E5M2 由第 1 步的 GPU 自测覆盖 |
| 矩阵分布 | random / normal / outlier | ✅ 题目要求「三种矩阵分别输出误差统计」 |

规模说明：**覆盖档用 256×256**，够把所有维度跑通又不至于写爆磁盘；**性能档用
1024×1024**，与 CPU 基线同规模，供报告的性能对比。

`test/run.sh`（nsys / ncu）是**尽力而为**：工具没装、或 ncu 缺 profiling 权限
（`ERR_NVGPUCTRPERM`）都只跳过并提示，不影响前面的正确性与性能数据。

## 产出

全部落在 `test/out/`（已被 `.gitignore` 忽略，不会误提交）：

| 路径 | 内容 |
|---|---|
| `summary.txt` | 环境信息 + 汇总表 + 每个组合的完整指标 |
| `<格式>/<输入>/metrics.json` | 误差 / 压缩率 / kernel 时间 / 有效带宽 / 对拍结果 |
| `<格式>/<输入>/metrics.log` | 同一份指标的人类可读版 |
| `<格式>/<输入>/weights.bin`、`dequant.bin` | 低精度权重（LPW1）与反量化输出 |
| `nsys/` | `kern_sum.txt`（各 kernel 耗时排序）、`mem_time_sum.txt`（拷贝耗时）、原始 `.nsys-rep` |
| `ncu/` | 每个 kernel 一份 `--set basic` 的 CSV |

把 `test/out/` 整个目录（或其中 `summary.txt` + `nsys/` + `ncu/`）发回即可用于撰写总结报告的
性能与分析章节。

## 手动跑

```bash
./test/test_all --repeat 10          # 全套（等价 make test-fast 的测试部分）
./test/test_all --outdir my_out      # 换输出目录
./test/test_all --repeat 3           # 少重复几次，快速冒烟

bash test/run.sh                     # 只跑 nsys / ncu
NO_PROFILING=1 bash test/run.sh      # 跳过
```

> 计时口径：`metrics.json` 里的 kernel 时间由 `cudaEvent` 夹住 kernel 本体、
> `repeat` 次取最小值。**ncu 的时间不能当作性能数据**（它锁频并多次重放 kernel），
> 只用于定位瓶颈——所以性能数字一律取自 `make test-fast` 的输出。
