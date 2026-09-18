# Core：正式程序

全部正式源码、构建定义、配置、CPU 参考实现与测试都在这里。`Core` 可以单独复制并重新构建；不要复制 `build`、`build-musa` 等构建缓存。

正式运行只通过一个入口：`Core/tools/quantize.py`。它调用编译出的两个后端可执行文件，按配置选择量化格式。

---

## 目录作用

| 目录 | 作用 |
|---|---|
| `app/` | 程序入口与文件任务：分发命令、执行量化/反量化流程、写误差日志 |
| `kernels/` | GPU 算子与设备端编码辅助 |
| `runtime/` | 显存、上传下载、kernel 启动与计时 |
| `common/` | 公共数据结构、随机数、输出类型转换、计时 |
| `io/` | 输入文件、packed 权重、反量化张量的读写 |
| `reference/` | CPU 参考实现（用于校验 GPU 结果），含冻结数据 |
| `benchmarks/` | 正式性能计时与统计 |
| `tests/` | 算法回归、文件与工具测试 |
| `tools/` | Python 入口与辅助脚本 |
| `configs/` | 两种格式的 TOML 配置 |
| `backends/musa/` | 摩尔线程 MUSA 后端，见其 [README](backends/musa/README.md) |
| `CMakeLists.txt` | 构建定义，用 `LP_BACKEND` 选择后端 |

两个后端可执行文件：`pipeline_mxfp8` 与 `pipeline_nvfp4`，由同一套源码按格式编译两次得到。

---

## 环境要求

CMake ≥ 3.18、CUDA Toolkit、C++17、Python ≥ 3.11。

`nvcc` 必须能被 CMake 找到。若报 `No CMAKE_CUDA_COMPILER could be found`，是 `nvcc` 不在 `PATH`，显式指定即可：

```bash
-DCMAKE_CUDA_COMPILER=$(which nvcc || echo /usr/local/cuda/bin/nvcc)
```

默认架构为 `sm_75`，自带 PTX，可在更高架构 GPU 上运行，但会触发 JIT，性能不具代表性。做性能测量时按实际显卡指定：

```bash
-DCMAKE_CUDA_ARCHITECTURES=89      # RTX 4090
-DCMAKE_CUDA_ARCHITECTURES=86      # RTX 3060
```

---

## 操作步骤

以下命令均在**项目根目录**执行。

### ① 构建

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
```

产物：

```
Core/build/pipeline_mxfp8
Core/build/pipeline_nvfp4
```

### ② 正确性检查

```bash
ctest --test-dir Core/build --output-on-failure
```

预期输出：`100% tests passed, 0 tests failed out of 9`。

| 测试项 | 检查内容 |
|---|---|
| `core_tools` | Python 工具行为 |
| `core_reduction` | 局部/全局最大值归约与 global_scale，含 warp 边界与极端值 |
| `core_device_math` | 设备端算术与 CPU 逐位对照 |
| `<格式>_pipeline_io` | 文件读写、类型组合、尾部哨兵 |
| `<格式>_pipeline_regression` | 对 CPU 参考实现的算法回归 |
| `<格式>_frozen_hashes` | 冻结输入与 golden 的 SHA256 校验 |

也可单独运行某项：

```bash
./Core/build/pipeline_mxfp8 --self-test          # 算法回归
./Core/build/pipeline_nvfp4 --self-test
ctest --test-dir Core/build -R mxfp8             # 只跑 MXFP8 相关
```

冻结文件只校验，不重新生成。关闭 `BUILD_TESTING` 会去掉算法自测代码，普通文件任务仍有 CPU 对照。

### ③ 生成输入

```bash
python3 Core/tools/quantize.py generate --dtype fp32 --rows 1024 --cols 1024
```

产物（终端打印实际路径）：

```
input/<月日>/<编号>.fp32        输入张量，带头部，行主序
input/<月日>/<编号>.fp32.json   形状、分布、种子与 sha256
```

`--dtype` 可选 `fp32` / `fp16`，`--distribution` 可选 `uniform` / `normal` / `outlier`。
同一编号可同时存在 `.fp32` 和 `.fp16`。

### ④ 量化并反量化

推荐直接指定参数：

```bash
python3 Core/tools/quantize.py run --input input/917/1.fp32 \
  --format mxfp8 --output-type fp16 --output-dir output/917 --save all
```

输入换成 ③打印的实际路径。`--format` 选择 `mxfp8/nvfp4`，`--output-type` 选择 `fp32/fp16/bf16`，`--output-dir` 指定实际保存目录，文件名为 `<输入名>_<格式>_<backend>_<入>_<出>.*`。重复运行自动编号。`--save` 默认 `all`，可改为 `weights`、`tensor`、`log` 或组合，如 `--save weights log`；只控制保存，完整计算和 CPU 校验仍执行。

以下配置文件用法继续支持，命令行显式参数优先于配置。省略配置时默认 `block/nearest/seed=1234`、FP32 输出；格式必须指定，block size 由格式决定。

配置在 `Core/configs/` 下。`output_type` 决定反量化输出类型（`fp32` / `fp16` / `bf16`）；融合路径使用 `scale_mode = "block"` 与 `rounding = "nearest"`。

```bash
python3 Core/tools/quantize.py run --config Core/configs/mxfp8.toml   --input input/916/1.fp32

python3 Core/tools/quantize.py run --config Core/configs/nvfp4.toml   --input input/916/1.fp32
```

把示例日期与编号换成 ③实际打印的路径。

省略 `--output-dir` 和 `--prefix` 时的产物：

```
output/<月日>/<编号>/<格式>/<backend>/<入>_<出>.lpq    低精度权重
output/<月日>/<编号>/<格式>/<backend>/<入>_<出>.<出>   反量化张量
output/<月日>/<编号>/<格式>/<backend>/<入>_<出>.json   误差与性能日志
```

`<backend>` 为 `cuda` 或 `musa`；`<入>_<出>` 形如 `fp32_fp16`。终端会打印三个文件的完整路径。
重复运行同一输入不会覆盖旧结果，而是分配新前缀。

附加参数：

```bash
--prefix /tmp/my-result     # 手动指定输出前缀
--output-dir output/917     # 手动指定文件夹，与 --prefix 二选一
--save weights log          # 仅保存权重及日志；默认 all 保存题目要求的全部文件
--output-type bf16          # 覆盖配置中的输出精度
--scale-mode tensor         # 覆盖缩放方式，block/tensor
--rounding stochastic       # 覆盖舍入方式，nearest/stochastic
--seed 1234                 # 舍入随机种子
--json                      # 终端输出单行 JSON，便于脚本解析
```

### ⑤ 独立反量化已有权重

不需要原始输入或 CPU golden，直接从 `.lpq` 恢复：

```bash
./Core/build/pipeline_mxfp8 --dequant-file \
  output/916/1/mxfp8/cuda/fp32_fp32.lpq \
  /tmp/restored.bf16 bf16
```

输出路径请取未使用过的名称；同名文件会被覆盖。最后一个参数可选 `fp32` / `fp16` / `bf16`。
NVFP4 换成 `./Core/build/pipeline_nvfp4`。

### ⑥ 误差评估

> 误差评估默认自动分配输出目录；性能测试显式指定的记录目录需要使用新名字，避免覆盖旧记录。

```bash
python3 Core/tools/quantize.py evaluate --backend cuda
```

产物：

```
input/evaluation-v1/{uniform,normal,outlier}.{fp32,fp16}   固定输入
input/evaluation-v1/manifest.json                         参数和SHA256
output/evaluation-v1/cuda/{uniform,normal,outlier}.json     分类误差统计
output/evaluation-v1/cuda/summary.json                     全部组合汇总
output/evaluation-v1/cuda/status.json                      完成状态
output/evaluation-v1/cuda/<分布>/<格式>/                    每组权重、反量化张量与日志
```

覆盖 3 种分布 × 2 种输入类型 × 2 种格式 × 2 种缩放 × 2 种舍入 × 3 种输出 = 144 组。
六份输入与每组的三类结果都持久保存。默认输出目录已存在时自动使用 `cuda-2`、`cuda-3`；显式传 `--directory` 时必须是**未存在**的新目录。`status.json` 中 `state=complete`、`completed=144` 表示整批完成，详见 [输出说明](../output/README.md)。

### ⑦ 性能测试

```bash
python3 Core/tools/benchmark.py --directory records/my-benchmark-01 --repeats 20
```

产物：

```
records/my-benchmark-01/RESULTS.md
records/my-benchmark-01/summary.json
records/my-benchmark-01/<格式>_<元素数>.jsonl
records/my-benchmark-01/environment.json
```

固定 FP32、`block+nearest`，测 1M / 4M / 16M，预热 3 次。正式优化必须 `--repeats 20` 并复用 `input/benchmark-v1/` 的冻结输入；试跑加 `--exploratory`。条件与加速比口径见 [固定协议](../docs/BENCHMARK_PROTOCOL.md)。

其他参数：

```bash
--backend musa                     # 换后端
--build-directory /path/to/build   # 自定义构建目录
--input-directory input/benchmark-v1
--exploratory                      # 允许非 20 次，不能作为正式轮次
```

### ⑧ 用 nsys 定位阶段

```bash
python3 Core/tools/profile.py --directory records/my-profile-01
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  records/my-profile-01/mxfp8.nsys-rep
```

产物：

```
records/my-profile-01/mxfp8.nsys-rep      nsys 原始报告
records/my-profile-01/nvfp4.nsys-rep
records/my-profile-01/mxfp8.sqlite        nsys 导出的数据库
records/my-profile-01/nvfp4.sqlite
records/my-profile-01/mxfp8_capture.log   采集过程日志
records/my-profile-01/nvfp4_capture.log
records/my-profile-01/mxfp8_stats.txt     kernel 汇总
records/my-profile-01/nvfp4_stats.txt
records/my-profile-01/version.txt
```

仅适用于 CUDA 后端。`--directory` 同样必须是未存在的新目录。
nsys 报告包含预热与内部对照，不能把累计占比当作单次默认流程占比。

### ⑨ GPU 检查

```bash
compute-sanitizer --tool memcheck ./Core/build/pipeline_nvfp4 --self-test
```

`--tool` 可换 `racecheck`、`synccheck`。MXFP8 同理。

---

## 修改 kernel 后的流程

1. 改 `Core/kernels/` 下的实现
2. `cmake --build Core/build -j 4`
3. `ctest --test-dir Core/build`（必须 9/9）
4. benchmark 保存到**新目录**
5. 更新根目录 `OPTIMIZATION_LOG.md`

加速比只列相对上一轮与相对本格式第 1 轮的倍数。

---

## 底层可执行文件

通常不需要直接调用。`python3 Core/tools/quantize.py` 已覆盖以下功能：

```text
文件任务: pipeline input packed output block|tensor nearest|stochastic fp32|fp16|bf16 block_size seed verify|noverify
反量化:   pipeline --dequant-file packed output fp32|fp16|bf16
性能测试: pipeline --benchmark elements repeats [fixed_input.fp32]
固定输入: pipeline --benchmark-export elements output.fp32
正确性:   pipeline --self-test [reference_directory]
```

`--benchmark-export` 用于重建冻结输入，用法见 [固定协议](../docs/BENCHMARK_PROTOCOL.md)。

---

## CPU 参考实现

`reference/mxfp8/` 与 `reference/nvfp4/` 定义两种格式的确定性结果，GPU 实现必须以它们为对照。各自的构建与验证步骤见其 README：

- [reference/mxfp8/README.md](reference/mxfp8/README.md)
- [reference/nvfp4/README.md](reference/nvfp4/README.md)

正常使用 CUDA 程序**不需要**单独构建这两个目录，CTest 已覆盖其冻结校验。

---

## 其他工具

```bash
python3 Core/tools/compare_benchmarks.py --output /tmp/cmp.md <a目录> <b目录>   # 比较两组结果
python3 Core/tools/optimization_report.py                                       # 生成 CUDA 优化报告
python3 Core/tools/cuda_optimization_report.py                                  # 生成累计效果报告
```
