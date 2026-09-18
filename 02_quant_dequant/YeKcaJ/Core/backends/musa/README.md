# 摩尔线程 MUSA 后端

在 MTT S4000（48 GiB、驱动/SDK 5.1.0、架构 `mp_22`）上完成真实 GPU 适配与两轮优化。构建后执行的是真实 MUSA kernel，CPU 仅作参考校验。

CUDA 后端不受影响，两者用不同的构建目录。

---

## 目录作用

| 文件 | 作用 |
|---|---|
| `include/cuda_runtime.h` | 把项目使用的内存、拷贝、event、错误接口名称映射到 MUSA（`cudaMalloc` → `musaMalloc` 等），只有 MUSA 构建包含此目录 |
| `include/cuda_fp16.h`、`cuda_bf16.h` | 引入 MUSA 的 FP16/BF16 类型与转换接口 |
| `quantize.cuh` | MUSA 融合量化 kernel：float4 加载、子组 shuffle 求最大值、一次 shared scale 广播 |
| `tools/` | 诊断工具，见其 [README](tools/README.md) |

公共代码（`kernels/`、`common/`、`runtime/`）在 MUSA 构建下通过条件编译复用，只有平台专有部分隔离在 `backends/musa/`。

---

## 环境要求

MUSA SDK（含 `mcc` 编译器，默认装在 `/usr/local/musa`）、CMake ≥ 3.18、Python ≥ 3.11。

先确认环境：

```bash
mthreads-gmi
/usr/local/musa/bin/mcc --version
```

---

## 操作步骤

以下命令均在**项目根目录**执行。MUSA 使用独立的 `Core/build-musa` 目录。

### ① 构建与检查

```bash
cmake -S Core -B Core/build-musa \
  -DLP_BACKEND=MUSA -DMUSA_ROOT=/usr/local/musa -DMUSA_ARCH=mp_22 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build-musa -j 4
ctest --test-dir Core/build-musa --output-on-failure
```

预期输出：`100% tests passed, 0 tests failed out of 10`。比 CUDA 多一项 `musa_shuffle`，用于实测设备的 shuffle 语义。

产物：`Core/build-musa/pipeline_mxfp8`、`Core/build-musa/pipeline_nvfp4`。

参数字段：

| 变量 | 含义 |
|---|---|
| `LP_BACKEND` | 必须为 `MUSA` |
| `MUSA_ROOT` | SDK 安装路径 |
| `MUSA_ARCH` | 设备架构；MTT S4000 为 `mp_22` |

### ② 生成输入

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 128 --cols 129
```

产物（终端打印实际路径）：

```
input/<月日>/<编号>.fp32
input/<月日>/<编号>.fp32.json
```

输入支持 FP32 和 FP16；FP16 先在主机展开为 FP32。不支持 BF16 输入。

### ③ 量化并反量化

也可以直接指定参数，无需配置文件：

```bash
python3 Core/tools/quantize.py run --backend musa \
  --input input/917/1.fp32 --format nvfp4 --output-type bf16 \
  --output-dir output/917 --save all
```

示例输入替换成实际路径。三类文件直接保存到 `output/917/`，前缀为 `1_nvfp4_musa_fp32_bf16`。`--save` 默认 `all`，可多选 `weights tensor log`；未选文件不留存，计算与 CPU 校验仍执行。未指定输出目录时沿用下方布局。配置文件入口继续支持，命令行显式参数优先：

```bash
python3 Core/tools/quantize.py run --backend musa \
  --config Core/configs/musa_mxfp8.toml --input input/916/1.fp32

python3 Core/tools/quantize.py run --backend musa \
  --config Core/configs/musa_nvfp4.toml --input input/916/1.fp32
```

产物（注意 `musa` 这一层，与 CUDA 结果分开放）：

```
output/<月日>/<编号>/<格式>/musa/fp32_fp16.lpq    低精度权重
output/<月日>/<编号>/<格式>/musa/fp32_fp16.fp16   反量化张量
output/<月日>/<编号>/<格式>/musa/fp32_fp16.json   误差与性能日志
```

`output_type` 可选 `fp32` / `fp16` / `bf16`。同一输入在 CUDA 与 MUSA 下写入不同子目录，互不覆盖。

### ④ 独立反量化已有权重

```bash
Core/build-musa/pipeline_mxfp8 --dequant-file \
  output/916/1/mxfp8/musa/fp32_fp16.lpq \
  /tmp/restored.bf16 bf16
```

NVFP4 换成 `pipeline_nvfp4`。输出路径取未使用过的名称。

### ⑤ 误差评估

```bash
python3 Core/tools/quantize.py evaluate --backend musa
```

读取 `input/evaluation-v1/` 的六份固定输入（缺失时自动生成），向 `output/evaluation-v1/musa/` 保存 144 组权重、反量化张量及日志，同时输出分类统计、总汇总和完成状态。重复运行使用 `musa-2`、`musa-3`，与 CUDA 结果隔离。详见 [输出说明](../../../output/README.md)。

### ⑥ 性能测试

```bash
python3 Core/tools/benchmark.py --backend musa \
  --directory records/my-musa-benchmark --repeats 20
```

固定 FP32、`block+nearest`，测 1M / 4M / 16M。`resident_gpu` 使用 MUSA event 计时。
产物：`RESULTS.md`、`summary.json`、`<格式>_<元素数>.jsonl`、`environment.json`。

> MUSA 的性能结论与 CUDA **不能互推**：带宽、线程模型与特殊函数吞吐都不同，必须各自测量。跨平台比较用 `compare_benchmarks.py`。

---

## 实现说明

设备报告 `warpSize=128`，但 SDK 的同步 shuffle 实际使用 **32-lane 逻辑子组**，已用 `tests/musa_shuffle.cu` 独立验证 `width=4/8/16/32`。这个差异不能靠设备报告的 warpSize 推断。

融合 kernel 的纯 shuffle 广播曾在零 scale / 极小值场景回归失败，因此正式实现保留一次 shared scale 广播。

当前除法和乘法均用 FP64 中间结果一次 RN 转回 FP32，以保持数值规则；已替代最初高开销的 `__fdiv_rn`。设备算术测试覆盖百万对输入。

---

## 限制

- NVIDIA 的 `nsys` / `ncu` 与 `Core/tools/profile.py` **仅适用于 CUDA**，不能用于 MUSA
- MUPTI 活动采集受 MT-Perf 硬件连接问题阻断
- 完整设备内存检查工具缺失，现有验证不能声称已覆盖全部内存与竞争问题
- MUSA 专用 profiler 与进一步性能调优待开展

阶段计时与 trace 工具见 [tools/README.md](tools/README.md)。
适配进度与验收标准见 [docs/PLATFORM_ADAPTATION.md](../../../docs/PLATFORM_ADAPTATION.md)。
优化数据见 [docs/MUSA_OPTIMIZATION_LOG.md](../../../docs/MUSA_OPTIMIZATION_LOG.md)。
