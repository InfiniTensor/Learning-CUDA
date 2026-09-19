# 壁仞 106M 原生 SUPA 后端

本目录是 Hadamard 项目的独立壁仞移植。原始 Warp32、可选均衡打包和可选小批量发射三个固定版本，分别在壁仞 106M 完成原生 SUPA 构建、1504 组变换验证、180 项 API 契约检查、14 项 CLI 拒绝检查和 12 项独立主机 BF16 舍入自检。相同矩阵跨版本不累加成新覆盖。小批量版本另完成固定留出矩阵的交错独立进程复测。

**正确性通过不等于某个方法在所有形状上更快：原始 Warp32 有明显退化，均衡打包不适合全局启用，小批量选项也只针对受测边界。**下文保留全部版本、负例、探索轮次和选择理由，API 默认仍为共享内存 `Optimized`。

壁仞当前设备为 warp32，天数和沐曦已经验证的 Warp64 路径不能直接套用。实现、测试、版本和原始结果均需独立归档，不继承其他平台的通过数量或速度比。

## 当前实机环境与资源边界

| 项目 | 已观察到的状态 |
|---|---|
| 设备 | 壁仞 106M，界面标称显存 32GB |
| 设备 API 可见显存 | 34,091,302,912 字节；保留该原始数值，不把标称容量和可见容量混为一列 |
| 执行资源 | 64 个 SM、warpSize=32、每 block 最大 1024 线程、共享内存 32 KiB |
| SDK | SUPA SDK `1.10.0.1.rc1` |
| 编译器 | `brcc 1.10.0`，Clang `16.0.1` |
| 设备管理工具 | BRSMI `1.10.2`、Driver `1.10.1`、SUPA `1.10` |
| 运行库 | 原生 `libsupa-runtime.so` |
| CUDA 兼容层 | 当前环境未找到 SUDA，不以 CUDA API 兼容层为前提 |
| Python 框架 | 当前 PyTorch 为 `2.8+cpu`，不能据此宣称已提供壁仞 GPU PyTorch 扩展 |
| 虚拟化显示 | SVI 显示 Disabled；该显示本身不能证明没有其他任务共享资源或已经获得独占整卡 |

SDK、驱动、设备工具和框架版本分别记录，不将其合并为一个推测版本。后续基准还需要记录采样时的设备状态、共享/独占证据及工具权限；不能只凭 SM 数、显存容量或 SVI 状态宣称独占与固定时钟。

## 原生 SUPA 路线

本平台使用真实 `brcc` 编译 `.su` 源文件，包含已安装 SDK 的 `supa.h`、`device/supa_fp16.h`、`device/supa_bf16.h`，链接本机 `libsupa-runtime.so`。项目源文件只引用厂商接口，**不将私人 SDK 头文件实现复制进公开仓库**。使用者需要单独安装相应 SDK。

运行时使用 SUPA 的 `suError_t`、`suStream_t`、`suEvent_t` 和 `su*` 函数；FP16/BF16 存储类型为 `float16`、`bfloat16`。这不是给 CUDA 函数名简单加宏，也不是将 CPU 结果作为设备输出。

迁移时已经核对的接口差异包括：

- 设备分配使用 `suMallocDevice`，不能推测成 `suMalloc`。
- `suMemcpyAsync` 的参数顺序为 `(dst, src, bytes, stream, kind)`，stream 位于方向枚举之前；不能沿用 CUDA 顺序。
- 使用 SUPA 自己的 stream/event 句柄和错误枚举，不与 CUDA 句柄混传。显式检查发射返回值和后续 stream 同步的执行错误。
- 初始化、上传、kernel、回读放在同一个显式 stream，不能假设 SUPA 与 CUDA 有相同的默认 stream 隐式依赖规则。
- 编译器使用真实 SUPA 宏 `__SUPACC__`，设备分支由 `__SUPA_ARCH__` 区分；不伪造 `__CUDACC__`、`__ILUVATAR__` 或 `__MACACC__`。

不能将天数的 `-x ivcore`、沐曦的 cucc/cu-bridge 或 NVIDIA 的 `-arch=sm_89` 命令照搬到本平台。最终复现命令应以独立 runner 的真实 brcc 及 include/lib 路径为准。

## BF16 最近偶数舍入：先保留失败，再修正语义

最初直接使用原生 BF16 转换时，实机样例输入 FP32 位模式 `0x3f818000`：独立最近偶数舍入参考期望 BF16 `0x3f82`，设备却给出 `0x3f81`，探针明确失败。该输入恰好位于两个 BF16 表示值中点，低候选尾位为奇数，因此按最近偶数规则应选高候选。

这条记录证明**该设备/SDK 的这一路原生转换不能在项目里直接当成已经满足 RNE 的操作**。不能把参考答案改成 `0x3f81`、放宽阈值或隐藏 BF16 来绕过失败，也不把一个反例泛化为所有未来 SDK 的行为。

修正使用本项目自行实现的位级 RNE helper：先从 FP32 位模式计算最近偶数的 BF16 表示，再得到一个 BF16 可精确表示的 FP32 值，最后转为 SUPA 的 `bfloat16` 存储。合法位转换使用固定大小的 `memcpy`，不通过违反别名规则的指针强转读取。此实现遵循格式定义，不复制厂商 SDK 的转换代码。

输入生成、设备变换输出、独立参考输出及融合量化的中间舍入必须统一到同一 RNE 契约；尤其不能用原生 BF16 主机构造器的不同舍入行为生成“参考”，否则设备和参考同时犯错也可能误判通过。最终仍须通过独立 FP64 数值参考与位模式/边界检查，而不是只比较两个共享同一路 helper 的实现。

原失败与修正探针两个独立源文件版本及编译/运行日志已保留在 [initial](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/initial/transfer_manifest.json)：[原始失败](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/initial/smoke_run.log)、[修正通过](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/initial/smoke_rne_run.log)。单独的探针不代表完整 Hadamard；后文提供随后独立运行的完整结果。这些结果仍**不证明硬件原生 BF16 算术加速**，实现使用 BF16 存储/转换与 FP32 内部加减。

## 最小实机探针的准确范围

- FP16、BF16 各 257 个样例，覆盖随机正负数、正负零、舍入中点和部分边界值。修正后的 BF16 存储位模式、随后转换为 FP32 的简单算术与独立宿主机参考一致。
- 设备缓冲区前后各 17 个元素的哨兵、输入保持和仅 2 字节对齐的半精度存储检查通过，覆盖不满线程块的尾部。
- 两个完整的 32-lane warp，对 XOR 距离 `1、2、4、8、16` 独立检查，共 320 个观察值通过。每个 warp 使用可区分的数据，防止跨 warp 取错值被相同输入掩盖。

这里的 shuffle 通过结果仅覆盖已运行的完整 warp/整数交换。它不等于浮点蝶形、多寄存器交互、尾行、跨 CTA 网格复用或融合量化已经通过。SUPA 接口名称带有 `_sync` 时，也不能未经核验就套用 CUDA 对活跃线程 mask 的保证；完整 warp 参与、无效逻辑元素填零和分支收敛仍需在实现中保证。

## 完整 API、复现入口与待验收项

独立接口见 [hadamard_api.h](hadamard_api.h)，实现见 [hadamard_api.su](hadamard_api.su)，命名空间为 `hadamard::biren`。声明提供 `transform`、`quantize_int4`、`transform_int4` 的 `float16`/`bfloat16` 重载，返回 `suError_t`，使用 `suStream_t`。方法为 `Baseline`、共享内存 `Optimized`、条件编译的 `Warp32`，默认仍为 `Optimized`；固定版本的三种方法均有完整正确性证据，性能结论则分别列出。

接口为原生 C++ 异步调用：连续 FP16/BF16 `[rows,n]`，N 为 1～256 的 2 次幂，内部 FP32；连续 `[batch,seq,heads,head_dim]` 在检查乘法溢出后展平成 rows，不接受 stride 参数。scale 必须有限且为正；输入应有限，变换输出应在存储类型的有限范围内。

INT4 目标语义为 `[-7,7]`，每行 `s=max(abs(x))/7`，全零行取 1，最近偶数舍入，偶数元素在低四位，N=1 的空高四位为零；融合路径先按 FP16/BF16 存储类型舍入再量化。

调用方提供 `rows*n` 个输入/变换输出元素、`rows*ceil(n/2)` 个 packed 字节及 `rows` 个 float scales。半精度指针要求 2 字节对齐、scales 要求 float 对齐；transform 只允许完全原位或互不重叠，量化各缓冲区互不重叠。API 检查参数、对齐、大小溢出和重叠，但不查询真实分配容量或设备归属。

所有发射都使用调用方 stream，不分配、不复制、不等待。调用方检查发射错误，并在使用回读结果前同步相应 stream。合法 `rows=0` 只检查 N、scale、方法后返回，不发射内核。

Warp32 需要真实 SUPA 编译环境和 `-DHADAMARD_BIREN_WARP32`，调用方还须在初始化时经 `suGetDeviceProperties` 确认当前设备 `warpSize==32`；该查询不在每次 API 发射路径内进行。未编译支持的非空 Warp32 调用返回 `suErrorNotSupported`，不静默回退。其他同为 warp32 的设备仍不能直接继承本平台语义或性能结论。

### 固定版本复现命令

runner 位于 [run_platform.py](run_platform.py)，原生验证程序为 [validate_and_benchmark.su](validate_and_benchmark.su)。复现下面原始 Warp32 记录时，先固定 `2cbaf41` 源码；从项目 `03_hadamard_tc/a962695448-rgb` 目录执行，结果目录必须是新目录。后续可选实验不自动继承本轮性能结果：

```bash
# 快速工具链/小矩阵检查，不代表完整验收。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --warp32 --quick --no-benchmark \
  --output results/biren/warp32_quick_01

# 完整矩阵及九路径基准。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --warp32 --repeats 100 --groups 5 \
  --output results/biren/warp32_full_01

# 如仅复测共享路径，省略 --warp32 并使用新目录。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --no-benchmark \
  --output results/biren/shared_full_01
```

`--sdk-root` 默认 `/usr/local/birensupa/sdk/latest`，`--compiler` 默认该 SDK 下的 `brcc/bin/brcc`；runner 根据实际解析的 SDK 目录读取 `supa/include` 与 `supa/lib`。编译参数为 `-x supa -std=c++17 -O2`、项目/SDK include 路径、两个 `.su` 文件以及 `-lsupa-runtime` 和相应 rpath；`--warp32` 额外定义项目专用宏。这里的 `--warp32` 是 runner 的构建选项，不是生成二进制的运行参数。

runner 仅为子进程设置 `SUPA_PATH=SDK/supa`、`BIREN_HOME=SDK`、工具搜索路径及实际存在的 SUPA/brcc 库目录，不修改用户系统环境或替换框架。设备库搜索需要前者指向 `supa` 子目录，不能误指整个 SDK 根目录。缺失编译器、构建失败、检查失败都应以非零退出码结束；结果目录已存在则拒绝覆盖。

本次 Warp32 完整模式实际完成 1504 组变换、180 项 API 契约检查、14 项 CLI 拒绝检查，另外有 12 项独立写出的主机 BF16 RNE 边界检查；九路径基准每轮 1350 条组样本。主机检查、探针观察值和相同矩阵重复运行不叠加为 GPU 覆盖。

输出包括构建日志、CLI 拒绝日志、验证 JSON/日志、可选基准 CSV/日志及 `run_summary.json`。摘要固定实际源文件 SHA256、Git HEAD/状态、SDK/编译器、命令和退出码；其中算法来源提交与本次实际壁仞测试提交是不同字段，不混为同一版本。

## 原始 Warp32 版本的完整实机证据

实际测试提交为 `2cbaf41d9f71a657ca2cca027a3083d9d497322d`，工作树干净，五个源文件 SHA256 已与该提交的 Git 对象逐一核对。原生 brcc 构建退出 0、构建日志为空；完整 runner 为 `PASS`，`quick=false`、`full_matrix=true`、`warp32_enabled=true`。源版本、命令、SDK 环境、原始日志与各阶段退出码见 [run_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/run_summary.json)。

| 项目 | 当前状态 | 原始证据 |
|---|---|---|
| 原生构建与探针 | 原始 BF16 反例保留，RNE 修正探针通过 | [初始清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/initial/transfer_manifest.json) |
| 三种方法、BF16 RNE 与独立参考 | 1504 组变换及 12 项主机格式检查通过 | [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/validation.json) |
| API、内存与 INT4 | 180 项 API、14 项 CLI，字节与 scale 精确一致 | [完整日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/validation.log) 及各 `invalid_*.log` |
| 同机三轮基准 | 4050 条原始组样本，三进程退出 0 | [repeat_runs.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/repeat_runs.json) |
| 复现与归档 | 固定源码、环境及逐文件 SHA256 | [公开清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/manifest.json) |

完整矩阵覆盖 N=1～256 的全部 2 次幂、普通行数 1/3/17/257、scale=1 和 `1/sqrt(N)`，N=1 不重复计算相同 scale。输入为均匀 `[-1,1)`、标准差 0.5 的正态分布、幅度 8 离群值、全零和单位脉冲；随机分布使用种子 123/8042/15961，确定性输入只计一次。另有 N=1/2、rows=65537 和 262145 的网格复用/尾行检查。

| 精度 | 实际变换组合 | 检查元素 | API 契约检查 | 已舍入参考最大绝对误差 | 未舍入参考最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| FP16 | 752 | 4,105,264 | 90 | 0.00390625 | 0.015620231628418 |
| BF16 | 752 | 4,105,264 | 90 | 0.000003814697265625 | 0.124984741210938 |

每 dtype 的所有元素在基线、共享优化、Warp32 及对应原位路径上逐字节一致；CPU 对实际设备输出量化、各方法分步与融合量化的全部 packed bytes 和 float scales 精确一致。262145 行 N=1/2 两例合计 786,435 元素，已经包含在表内，不重复加总。

完整验证沿用项目已舍入参考：独立 FP64 稠密 Hadamard 结果经 FP32 转换后，按输出 dtype 的 RNE 舍入再比较，FP16 绝对误差严格小于 `1e-2`、BF16 严格小于 `5e-2`。**阈值判定仅针对已舍入列；未舍入误差包含最终存储舍入，不能也说它低于该阈值。**主机 BF16 参考使用独立高/低位比较及手写边界，不调用设备 helper 或将原生不同舍入当成预期。

### SDK 路径错误的失败记录

最初的 `8040b60` quick 构建因设备库搜索路径错误而失败，brcc 报无法找到 `libsupadevice`；[原始构建日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/diagnostics/runs/biren-api-quick-8040b60/build.log) 和 [运行摘要](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/diagnostics/runs/biren-api-quick-8040b60/run_summary.json) 均保留。修正为 `SUPA_PATH=SDK/supa`，使编译器找到该 SDK 的设备库，随后 `2cbaf41` 成功构建并完成全部运行。没有使用关闭设备库链接的参数绕开问题，先前失败也不计为通过。

### 同机三轮性能：保留 Warp32 退化

三轮使用同一二进制 SHA256 `aa7e015ef5ce8a991c3431ef1c299a1529da4d093028163029a5dd4aaf641901`。两 dtype × N=64/128/256 × rows=1/17/257/4096/16384，共 30 个 shape/dtype 条件；基线、共享、Warp32 各 transform/split/fused 三种操作，共九路径。每路径先预热 10 次，五组、每组重复 100 次，组间轮换方法顺序，每轮 1350 条组样本。

同轮相同 shape/dtype/scale/操作先取五组中位数再匹配比较：共享优化版有 **64/90** 组每轮比基线减少至少 5% 耗时，Warp32 有 **81/90** 组。Warp32 相对基线有六组稳定退化，全部是 rows=17 的 transform；相对共享版有 29/90 组更慢，只有 49/90 组每轮达到至少 5% 降时。因此默认仍保留共享 `Optimized`，不能因使用 warp shuffle 就声称更快。

| 操作 | 基线 / 原始 Warp32 | 共享优化版 / 原始 Warp32 |
|---|---:|---:|
| transform | 0.87596～1.95562 倍 | 0.85528～1.84059 倍 |
| split | 1.08715～2.07062 倍 | 0.83266～1.69125 倍 |
| fused | 1.02949～2.99272 倍 | 0.51151～1.76663 倍 |

比值低于 1 表示 Warp32 更慢。最明显的已记录退化是 FP16 `[16,64,16,256]` fused：Warp32 约 1416～1418 微秒，共享版约 725 微秒，慢约 95.32%～95.50%。负例与其他结果完整保存在三轮 [第一轮 CSV](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/benchmark.csv)、[第二轮 CSV](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/benchmark_run2.csv)、[第三轮 CSV](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/benchmark_run3.csv) 及 [中性事件描述分析](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/analysis/analysis.json)。

Warp32 内部分步/融合比为约 1.600～2.089 倍，融合降时约 37.52%～52.13%。**内部融合有效不代表胜过共享实现**；不能用这个比值隐藏上面的跨方法退化。所有范围只覆盖当前 N=64/128/256 的实测性能矩阵，不外推 N=1～32。

### 属性诊断与后续实验边界

独立属性程序及日志保存在 [diagnostics](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/diagnostics/manifest.json)，它查询 kernel 元数据，不执行性能测量。旧版及后续 balanced-pack 实验的 N=256 Warp fused 均报告 `numRegs=72`、`localSizeBytes=0`，共享版报告 48 个寄存器。**不能据此说已发生寄存器 spill，也不能把寄存器差异当成性能退化的已证实因果。**

下面分别报告可选 balanced-pack 和小批量发射实验。它们具有独立源码提交、编译开关和原始记录；不能将某一路的局部改善扩展为其他路径或其他版本的收益。

## 可选均衡打包实验：保留局部改善与负例

固定提交 `c01ac87fed92dc5d9bdf8bcf27f3dc9243a2818f` 的 `--balanced-pack` 仅改动 Warp32 的 INT4 打包路径，影响 split/fused，不改变 transform。完整运行为 PASS：1504 组、180 API、14 CLI、12 主机 BF16 格式检查；具体源码和二进制 hash 见 [run_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/run_summary.json)，实际计数及误差见 [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/validation.json)。

本实验保存了 **五轮共 6750 条原始组样本**。第 2/3 轮与属性程序的 CPU 编译共同启动，在查看这些结果前已记录排除决定，主统计选用第一次 `benchmark.csv` 和新的 `benchmark_run4.csv`、`benchmark_run5.csv`，共 4050 条。原始第 2/3 轮完整保留，[timing_selection_note.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/timing_selection_note.json) 记录提前决定的原因；不能说本实验总共只有 4050 条，也不能只发布挑出的快样本。

同轮形状/精度/scale 匹配后，主三轮的基线/候选比为 split 约 **1.104～2.180 倍**、fused **1.020～2.784 倍**；共享优化版/候选则分别为 **0.837～1.653 倍**、**0.548～1.740 倍**。低于 1 表示候选更慢，不能据融合有效就宣布全面胜过共享版。

与旧版不同时间段的阶段中位数对照，FP16 rows=16384、N=256 fused 从约 1417.44 降至 1323.65 微秒，减少 6.62%；该条件的基线和共享控制路径分别变化约 +0.11% 和 -0.03%。但是候选仍比本轮共享版慢约 82.3%～82.5%。FP16 rows=257、N=128 fused 阶段中位下降约 16.74%，按共享控制归一后约 14.80%。

负例包括 BF16 rows=257、N=64 的阶段耗时增加约 8.63%，以及 FP16 rows=16384、N=128 增加约 7.18%。60 个量化操作比较中，42 个阶段中位更快、18 个更慢，其中七个退化超过 3%。**旧新版本不是同一时间段的同进程 A/B，这些是观察描述，不能当成已排除时间漂移的因果估计。**transform 未改，不能把它的波动计为改动收益。

结论是保留该可选实验供复现，不建议全局开启。全部五轮 CSV/日志、主样本选择和派生分析都在 [balanced_c01ac87 清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/manifest.json)；[分析](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/analysis/analysis.json) 明确引用 1/4/5。属性记录中旧新版本 N=256 Warp fused 都是 72 个寄存器、local=0，不能据此声称修复了 spill。

## 可选小批量发射与留出验证

固定提交 `8f75553a074d79b850294377d5aea6381e93da19` 的 `--small-batch-warp` 只在 `rows<=64` 时把 Warp32 发射改为每 CTA 一行、32 线程；超过阈值保持原发射。它可影响 Warp32 的 transform/split/fused，基线和共享实现不变。本次没有同时开启 balanced-pack。

完整矩阵独立通过 1504 组变换、180 API、14 CLI 和 12 主机 BF16 格式检查，三轮 4050 条组样本均成功结束。每 dtype 752 组、4,105,264 个元素，与基线、共享、原位和 INT4 参考一致；误差定义与前文保持一致。原始记录见 [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/small_8f75553/validation.json)、[运行摘要](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/small_8f75553/run_summary.json) 和 [分析](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/small_8f75553/analysis/analysis.json)。

为检查阈值附近而不是只看已有调优点，另固定 `rows={32,63,64,65}`、`N={64,128,256}`、两 dtype，共 24 个 shape/dtype 条件。rows=65 是不触发新逻辑的阈值外控制点。每条件执行两种二进制的 22 个输入参数组合，共 **528 个不同的输入参数组合、48 个验证进程**，全部通过；重复执行的 API/主机自检不另算新增覆盖。

性能复测为三轮，每个条件交错运行旧/新两个独立进程，按预先固定的轮次/条件奇偶规则轮换顺序，共 **144 个独立基准进程**。这不是同进程 A/B。每个条件仍为九方法、五组、100 次重复，六份旧/新每轮合并 CSV 各有 1080 行，共 **6480 条不同的原始组样本**；合并 CSV 只是底层 raw CSV 的拼接，不算第二份新观察。

[holdout_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/holdout_summary.json) 记录 PASS、固定矩阵、交错顺序、全部命令/退出码和来源文件。旧二进制 SHA256 与 `2cbaf41` 的构建摘要匹配，新二进制与 `8f75553` 匹配，运行前后 hash 未变；因此不仅依赖脚本调用方给出的源码标签。实验脚本本身不编译、不调参、不改变矩阵，调用方负责避免并发编译或 GPU 工作。

原始与分变体分析已归档：[旧版分析](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/analysis_old/analysis.json)、[新版分析](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/analysis_new/analysis.json)。独立复核核对了 391 份原始文件的 hash、旧新源码/二进制链及所有进程退出码；跨变体按相同轮次、形状、精度和操作直接配对，不将各自对 baseline 的比值当成旧新收益。

### 小批量留出结果与明确推荐范围

受新发射逻辑影响的 rows=32/63/64、三种 N、两 dtype、三操作，共 **54 个操作条件**。它们相对旧 Warp32 在三轮中全部更快，其中 **52/54** 每轮均减少至少 5% 耗时。最弱的 FP16 rows=63、N=64 fused 三轮只减少约 **3.946%～4.072%**；最强的 FP16 rows=63、N=256 fused 减少约 **53.120%～53.138%**。因此不能将 54/54“更快”写成 54/54“至少 5%”。

rows=65 不触发新发射逻辑，是阈值外控制。控制条件没有稳定达到每轮 5% 的变化，观察到的最差变化约为 **-1.541%**，不计作新方案功劳。这是交错独立进程的有限测量，不是已经排除所有系统波动的同进程因果实验。

与共享 `Optimized` 比较时，受影响的 54 个条件中仅 **47/54** 三轮更快，只有 **23/54** 每轮减少至少 5%。其余七个负例全部在 N=256，最差仍比共享版慢约 **9.825%**。小批量方案改善了旧 Warp32，不意味着它成为所有条件的最快方法。

**目前最明确的推荐范围是已测 rows=32/63/64、N=64、FP16/BF16 的 fused 路径**：相对同轮共享实现每轮耗时减少约 **10.09%～12.68%**。在本机与已验证 SDK 条件下，可显式编译小批量选项并选择 `Method::Warp32` 使用该路径；不据此改全局默认，也不扩展到未测 N/rows/设备。

主矩阵另外保留非交错阶段对照：rows=17 的六个 transform 条件，旧版三轮均比基线慢、新版三轮均比基线快，旧新阶段描述的耗时减少约 **7.773%～21.645%**。这不是前述留出实验的交错配对，不能直接作为同进程因果结论。rows=1 没有稳定 5% 收益，部分观察退化约 **2.44%**，所以不能宣传“所有小批量都更快”。

大输入 rows=16384、N=256 fused 不触发小批量逻辑，仍明显慢于共享版：FP16 慢约 **95.5%～95.7%**，BF16 慢约 **84.3%～84.6%**。这里继续选择共享实现更符合已取得的证据；balanced-pack 也不建议全局开启。

### 复现实验开关与固定留出脚本

两个可选开关必须与 `--warp32` 一起使用。下面的性能实验分别构建；两开关同时开启的组合已额外完成正确性检查，但没有组合性能测量，不能把任一单独实验的速度比套给组合：

```bash
# 固定 c01ac87 源码，复现均衡打包实验。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 \
  --warp32 --balanced-pack --output results/biren/reproduce_balanced_01

# 固定 8f75553 源码，复现小批量发射实验。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 \
  --warp32 --small-batch-warp --output results/biren/reproduce_small_01
```

自写留出脚本 [biren_small_batch_holdout.py](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/biren_small_batch_holdout.py) 的 SHA256 为 `614db3a3b32fc98708b02122a89d1dd42d4341e9308c68cc7d98f1c5b31fe9cc`，与实机摘要匹配。先分别用固定源码构建旧版和小批量版可执行文件，并保存构建来源；下例中的两个路径必须对应这两次构建的真实文件：

```bash
python3 results/biren/holdout_8f75553/biren_small_batch_holdout.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 \
  --old-binary results/biren/reproduce_original_01/validate_and_benchmark \
  --new-binary results/biren/reproduce_small_01/validate_and_benchmark \
  --old-source-id 2cbaf41d9f71a657ca2cca027a3083d9d497322d \
  --new-source-id 8f75553a074d79b850294377d5aea6381e93da19 \
  --output results/biren/reproduce_holdout_01
```

`--old-source-id`/`--new-source-id` 只是调用方标签，脚本不会凭标签验证源码。复现者必须另行核对二进制 hash、构建记录和源码提交。新结果目录不得已存在，脚本保留所有进程的原始 JSON/CSV/log，不搜索表现最好的参数。

## 最终编译组合检查：正确性与性能分别记录

同一固定源码 `8f75553a074d79b850294377d5aea6381e93da19` 另外执行两种构建配置，测试时工作树干净，两套各五个源文件 hash 均与该提交匹配：

| 构建配置 | 实际验证范围 | API 与专门检查 | 性能测量 |
|---|---|---|---|
| `--warp32 --balanced-pack --small-batch-warp` | 完整 1504 组通过，`quick=false`、`full_matrix=true` | 180 API、14 CLI、12 主机 BF16 RNE 检查通过 | 未测量 |
| 不启用上述三个开关 | **仅 quick 50 组通过**，`quick=true`、`full_matrix=false` | 122 API、12 项未编入 Warp32 路径检查、12 主机 BF16 RNE 检查通过 | 未测量 |

组合构建的二进制 SHA256 为 `776b7784f2420808c63932cbe871bfcc002e92e23c373622ef579dd46418aa27`，原始记录见 [组合验证](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/combined/validation.json) 和 [组合运行摘要](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/combined/run_summary.json)。默认无特性开关的二进制 SHA256 为 `eb022b93b6da249096e1e59069e7a300d29c782cea8edaa186a334ce7a4b43a9`，记录见 [默认 quick 验证](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/default/validation.json) 和 [默认运行摘要](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/default/run_summary.json)。默认构建也执行了 runner 的 CLI 拒绝检查；它与其他构建的重复检查不累计为新覆盖。

这确认了组合路径在现有完整矩阵中的正确性，以及默认构建的快速检查；**不能称默认无开关构建已独立跑过全量，也不能声称组合性能优于单独选项**。相同输入和重复 API/主机检查不叠加成更多独立用例，本次没有新增计时样本。

复现时固定相同源码，使用新的结果目录：

```bash
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 \
  --warp32 --balanced-pack --small-batch-warp --no-benchmark \
  --output results/biren/reproduce_combined_contract_01

python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --quick --no-benchmark \
  --output results/biren/reproduce_default_quick_01
```

[final_jobs_and_source.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/final_jobs_and_source.json) 保存采集时已经结束的作业退出码与源码状态，包括早期保留的失败，不能解释为“历史所有作业均成功”或实例释放回执。此批 39 份原始文件通过传输 hash 核验，自写脚本与两套构建/验证日志均收录在 [独立清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/contract_variants_8f75553/manifest.json)。

本次性能使用原生 SUPA event，区间排除分配、主机/设备复制、预热和验证；只读输入重复使用可能受缓存影响。事件区间可能含主机发射造成的空闲，不是端到端延迟或隔离测得的纯内核时长。逻辑读写量换算的 GB/s 不是实测物理显存带宽。SVI Disabled 不能证明没有共享干扰，少数重复运行也不是性能保证。

[原始版本清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/manifest.json) 保留 29 份完整运行原始文件和七份初始环境/探针的传输校验；[均衡打包](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/manifest.json)、[小批量版](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/small_8f75553/manifest.json)、[固定留出](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/manifest.json) 分别核验 39、29、391 份传输原始文件，并保留派生分析和自写脚本。诊断另有独立清单。全部目录 `.gitattributes` 使用 `-text`，逐字节保存日志和 CSV；测试源码 hash 对应固定 Git 提交，不能由后续工作树修改覆盖。

## 毫秒单位导出与有效数字

题目要求以毫秒汇报。原始 CSV 和本文中的微秒读数保留不变，五个分析目录分别增加 `method_summary_ms.csv`：

- [原始 Warp32 毫秒表](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/2cbaf41/analysis/method_summary_ms.csv)
- [均衡打包毫秒表](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/balanced_c01ac87/analysis/method_summary_ms.csv)
- [小批量版本毫秒表](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/small_8f75553/analysis/method_summary_ms.csv)
- [留出旧版毫秒表](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/analysis_old/method_summary_ms.csv)
- [留出新版毫秒表](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/holdout_8f75553/analysis_new/method_summary_ms.csv)

转换仅将所有以 `_us` 结尾的时间字段按十进制除以 1000，并将字段名改为 `_ms`；其他字段、列顺序和数据行顺序全部保留。**这是纯单位换算，不增加测量有效位数、精度或原始样本数。**例如 1417.44 微秒等于 1.41744 毫秒，不代表新增了一次更精确测量；百分比、比值、CV、样本计数等不换算。

自写 [convert_summary_ms.py](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/biren/diagnostics/convert_summary_ms.py) 使用十进制表示直接变更指数，不经过二进制浮点，也不覆盖原表或既有导出。可在项目目录执行：

```bash
python3 results/biren/diagnostics/convert_summary_ms.py \
  results/biren/2cbaf41/analysis/method_summary.csv \
  results/biren/balanced_c01ac87/analysis/method_summary.csv \
  results/biren/small_8f75553/analysis/method_summary.csv \
  results/biren/holdout_8f75553/analysis_old/method_summary.csv \
  results/biren/holdout_8f75553/analysis_new/method_summary.csv
```

仓库已包含导出结果，脚本遇到已有 `_ms.csv` 会拒绝覆盖；重现转换时应在独立临时副本中使用尚无该导出的目录。导出表和脚本均纳入对应目录的 SHA256 清单。

公开交付只包括项目自行编写的源代码、复现入口、环境说明及脱离私人访问凭据的实验结果。私人 SDK 头文件、实例访问地址和租赁凭据不进入仓库。本平台按固定源码和原始日志核对实现与测试结果；PR 和课程登记在项目所有者验收具体代码及证据后单独执行。
