# 天数智芯 COREX 后端

此目录提供天数智芯 MR-V100 的原生 COREX C++ Hadamard 与融合 INT4 后端。最新实测源码 **`a387db3332c6f9b01f128dd681848260c9691281`** 已完成包含 Warp64 的 **1504 组变换验证、180 项 API 契约检查、14 项 CLI 拒绝检查和三轮 4050 条基准样本**。推荐在实测支持条件下显式选择 `Method::Warp64`；API 默认仍为共享内存 `Method::Optimized`。

下方保留初版、共享内存版和 Warp64 版的独立版本与原始证据，测试数量不跨版本累加。支持范围是当前 MR-V100/COREX 组合，未宣称所有国产设备、原生 BF16 硬件算术、矩阵单元加速或所有可能输入上的全局最优性能。

## 首轮原型：构建与运行

在项目的 `03_hadamard_tc/a962695448-rgb` 目录执行，使用与驱动匹配的完整 COREX 开发环境：

```bash
mkdir -p build/iluvatar
/usr/local/corex/bin/clang++ -x ivcore -std=c++17 -O2 \
  -Iinclude platforms/iluvatar/shared_baseline.cu \
  -L/usr/local/corex/lib -Wl,-rpath,/usr/local/corex/lib -lcudart \
  -o build/iluvatar/shared_baseline
./build/iluvatar/shared_baseline
```

若供应商镜像使用其他 SDK 路径，调整路径。不要照搬 NVIDIA 的 `-arch=sm_89`；如需要指定天数目标架构，应使用该 SDK 和实机确认过的编译参数。

## 首轮原型：算法与验证范围

- 每个 block 处理一行，线程数为 `max(64, N)`，`N` 为 1～256 的 2 次幂。
- 全部蝶形加减使用 FP32；共享内存上每层先读取、全 block 同步、写回、再次同步。所有线程参加同步，不依赖 32 或 64 的 warp 宽度，也不依赖 WMMA。
- 变换后先按输出 FP16/BF16 舍入，再量化。每行对称 INT4 为 `[-7,7]`，采用最近偶数舍入，偶数元素放低四位；零行 scale 为 1，`N=1` 的空高四位为 0。
- 所有传输和 kernel 使用同一个显式非阻塞 stream，回传后同步，再在 CPU 上检查。
- 每种精度 162 组变换用例：9 种 N、3 种行数（1/17/257）、2 种 scale（1 和 `1/sqrt(N)`）、零值/脉冲/确定性随机输入。`N=1` 的两种 scale 数值相同，但两条参数组合均执行。
- 每组检查所有元素的 CPU FWHT 误差，并对首行/中间行/末行使用独立 FP64 稠密 Hadamard oracle；阈值严格小于 FP16 `1e-2`、BF16 `5e-2`。
- 对真实 GPU 变换输出运行 CPU 量化，检查全部打包字节和 scale 与 GPU 分步、融合路径完全一致；另有每种精度一组正负半整数最近偶数舍入测试。
- 随机输入在 `[-0.25,0.25)`；这只是初步移植范围，不等同于主线更广分布、离群值、多随机种子的完整验收。

输出包含真实设备名称、warp 宽度、运行时版本、驱动版本、每种精度用例数、元素数、最大绝对误差；失败以非零退出码退出。主程序始终尝试两种精度，不通过静默跳过 BF16 制造通过结果。

## 首轮原型：已取得的 MR-V100 实机证据

2026-09-05 在真实天数智芯 MR-V100 32GB 上完成了首轮 COREX 原生编译与运行，编译和验证进程均以退出码 0 结束。设备报告 `warp=64`、`runtime=10020`、`driver=10020`；后两个数值是兼容运行时 API 返回值，不代表使用 NVIDIA CUDA 10.2 编译器。实机编译器为 `/usr/local/corex/bin/clang++` 18.1.8，使用 `-x ivcore -std=c++17 -O2 -Iinclude -lcudart`。

| 精度 | 变换用例 | 检查元素 | 最大绝对误差 | 半整数舍入 | CPU / 分步 / 融合 INT4 |
|---|---:|---:|---:|---|---|
| FP16 | 162 | 843,150 | 0.00385761261 | 通过 | 全部字节与 scale 一致 |
| BF16 | 162 | 843,150 | 0.0301055908 | 通过 | 全部字节与 scale 一致 |

此处误差是 GPU 输出与**未舍入到 FP16/BF16 的 FP32 CPU FWHT，以及抽样 FP64 稠密 oracle**的最大绝对差。这个定义包含最终存储精度的舍入误差，不能直接与其他实验中采用已舍入参考输出的最大误差横向比较。

通过结果证明本程序的 BF16 存取/转换与 FP32 累加路径在该实机和 SDK 上可用；不证明硬件原生 BF16 算术，也不意味着国产平台适配已全部完成。

首轮源码的汇总 `printf` 触发 COREX 格式检查警告。最终版改用主机 C++ 输出流，算法和用例保持不变，并重新在 MR-V100 编译运行：编译无警告、退出码 0，324 组变换及两组舍入检查再次通过。最终源码 SHA256 为 `d8b00f6e60feaf0dd450ca66255e05bd7e0e01a060d911a9fc7c61779023f180`。

原始验证输出、空的成功编译日志、设备环境与哈希清单保存在 [results/iluvatar](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/manifest.json)。重复运行同一批组合不计为新增独立用例。

## C++ API 与共享内存优化实现

本节描述上述 `b326789` 实机版本的 `Baseline` 和 `Optimized` 两种方法；后续增加的方法必须单独验证，不能继承本轮结果。

| 文件 | 职责 |
|---|---|
| [hadamard_api.h](hadamard_api.h) | FP16/BF16 变换、INT4 量化及融合变换量化的公开接口和内存契约 |
| [hadamard_api.cu](hadamard_api.cu) | 基线、共享内存优化版，以及显式编译启用的 Warp64 实现 |
| [validate_and_benchmark.cu](validate_and_benchmark.cu) | 独立 FP64 稠密参考、接口检查、内存边界检查及设备事件计时 |
| [run_platform.py](run_platform.py) | COREX 构建、CLI 拒绝测试、正确性验证、基准汇总及文件哈希记录 |

`Baseline` 保留每个元素对应一个线程、每层两次同步和线程 0 串行最大值归约。`Optimized` 让每个线程独占一个蝶形的两个输入/输出，每层仅在结束时同步，并用并行最大值归约计算量化 scale。两者均不依赖 NVIDIA warp32 或天数 warp64 的隐式同步，不包含 WMMA。`Optimized` 是实现名称；是否更快由同机测量决定，不能由名称推断。

API 使用 `hadamard::iluvatar` 命名空间，提供 `__half` 和 `__nv_bfloat16` 重载：

- `transform`：Hadamard 变换，允许完全原位操作 `input == output`；不允许其他部分重叠。
- `quantize_int4`：量化已经舍入到 FP16/BF16 的变换输出。
- `transform_int4`：融合变换与量化，变换后先舍入到对应存储类型，保持与分步路径相同的量化语义。

输入为连续设备内存 `[rows, n]`，`n` 是 1～256 的 2 次幂。连续 `[batch, seq, heads, head_dim]` 按 `rows=batch*seq*heads`、`n=head_dim` 传入；乘法溢出由调用方先检查。API 不接受 stride 参数，也不会为非连续视图自动整理内存。`scale` 必须有限且为正；输入应有限，变换结果应在输出存储类型的有限范围内。

调用方负责分配当前设备上容量足够的缓冲区：输入/变换输出各 `rows*n` 个对应 dtype，INT4 输出 `rows*ceil(n/2)` 字节，scale 输出 `rows` 个 float。输入和变换输出要求 2 字节对齐，scale 要求 float 对齐；量化接口的各缓冲区互不重叠。API 检查形状、scale、方法、对齐、整数/地址范围溢出及缓冲区重叠，**不查询真实分配容量或设备归属**。

全部操作在调用方 stream 上异步发射，不分配、不复制、不等待。返回值用于检查参数/发射错误，执行错误由调用方同步 stream 时检查。`rows=0` 合法：检查 `n`、`scale`、方法后返回，不检查指针或发射内核；CLI 为便于实际测试，只接受正形状。

调用示例（缓冲区及 stream 已由调用方创建）：

```cpp
#include "platforms/iluvatar/hadamard_api.h"

cudaError_t transform_then_quantize(
    const __half* input, __half* output, std::uint8_t* packed,
    float* scales, std::size_t rows, cudaStream_t stream) {
    namespace api = hadamard::iluvatar;
    auto status = api::transform(input, output, rows, 128, 1.0f, stream,
                                 api::Method::Optimized);
    if (status != cudaSuccess) return status;
    return api::quantize_int4(output, packed, scales, rows, 128, stream,
                              api::Method::Optimized);
}
```

只需要量化输出时，可用 `transform_int4(input, packed, scales, rows, 128, 1.0f, stream, method)` 替换两次调用。不要在前一调用结束前释放缓冲区；读取主机结果前还需安排复制及 stream 同步。这是原生 C++ API，当前不提供天数 PyTorch Python 扩展。

## 新版构建与可复现运行

从项目 `03_hadamard_tc/a962695448-rgb` 目录执行。runner 只需要 Python 标准库和已配套安装的 COREX SDK，不安装或替换框架、驱动。结果目录必须尚不存在，避免覆盖先前证据。

下面不带 `--warp64` 的命令重现共享内存路径；MR-V100 的推荐 Warp64 命令见后文专节。选择测试版本时同时固定相应源码 commit，不将不同版本样本混成一次实验。

```bash
# 快速检查工具链和小矩阵；即使通过，也不代表完整验收。
python3 platforms/iluvatar/run_platform.py \
  --quick --no-benchmark --output results/iluvatar/api_quick_01

# 正式矩阵 + 默认基准：不要加 --quick。
python3 platforms/iluvatar/run_platform.py \
  --repeats 100 --groups 5 --output results/iluvatar/api_full_01
```

默认编译器是 `/usr/local/corex/bin/clang++`，SDK 根目录是 `/usr/local/corex`。可通过 `--corex-root /实际/SDK/路径` 或 `--compiler /实际/编译器/路径` 指定；二者应属于匹配的安装。runner 使用 `-x ivcore -std=c++17 -O2`，链接配套 `libcudart`；不使用 NVIDIA `-arch=sm_89`，也不启用 fast-math。

需要仅构建可执行程序时：

```bash
mkdir -p build/iluvatar
/usr/local/corex/bin/clang++ -x ivcore -std=c++17 -O2 -Iinclude \
  platforms/iluvatar/hadamard_api.cu \
  platforms/iluvatar/validate_and_benchmark.cu \
  -L/usr/local/corex/lib -Wl,-rpath,/usr/local/corex/lib -lcudart \
  -o build/iluvatar/validate_and_benchmark

# 直接运行默认完整矩阵。
./build/iluvatar/validate_and_benchmark --validate \
  --json build/iluvatar/validation.json

# 定位一个连续四维形状；这是定向验证，full_matrix 为 false。
./build/iluvatar/validate_and_benchmark --validate --dtype bf16 \
  --batch 2 --seq 17 --heads 3 --dim 128 \
  --json build/iluvatar/validation_custom.json

# 已构建程序的独立基准进程；每轮使用新的文件名。
./build/iluvatar/validate_and_benchmark --benchmark --groups 5 --repeats 100 \
  --csv build/iluvatar/benchmark_run02.csv
```

直接执行二进制会按指定路径写文件，应使用新文件名；只有 runner 对整个结果目录实施“不覆盖既有目录”的检查。二进制参数错误返回 2，设备/正确性失败返回 1；runner 只有请求的所有阶段及对应退出码检查成功才返回 0。快速模式和 `--no-benchmark` 下的成功都不能当作完整性能验收。

runner 的主要输出：

| 文件 | 内容 |
|---|---|
| `build.log`、`invalid_*.log` | 编译原始输出、CLI 拒绝测试原始输出 |
| `validation.log`、`validation.json` | 设备信息、进度、实际用例数、误差与 API 契约检查数；JSON 标明 `full_matrix` |
| `benchmark.log`、`benchmark.csv` | 基准进度和每组未经筛选的事件计时样本 |
| `run_summary.json` | 源文件 SHA256、Git HEAD/状态、编译器信息、各阶段命令/退出码、结果摘要及产物哈希 |
| `validate_and_benchmark` | 本轮编译的本机可执行文件；其 SHA256 单独记录 |

`run_summary.json` 中的 Git HEAD 是执行时读取值；若源码尚未提交，必须同时依据 `git_status` 和源文件 SHA256 追溯，不能仅将 HEAD 当成实际测试代码。发布日志前核查访问地址等私有信息；包含私有信息的原件与本机二进制可单独保存，公开副本使用自己的哈希。

## 共享内存 API：完整验证矩阵

以下矩阵已由固定版本 `b326789` 完成，实际计数与误差见后文原始结果：

- FP16/BF16 分别遍历 `N={1,2,4,8,16,32,64,128,256}`、`rows={1,3,17,257}`，scale 为 1 和 `1/sqrt(N)`；N=1 的重复 scale 只计算一次。
- 输入包括均匀分布 `[-1,1)`、标准差 0.5 的正态分布、带幅度 8 离群值的均匀分布、全零和单位脉冲。前三种使用种子 `123、8042、15961`；零值/脉冲没有随机性，只计一次。
- 全量模式另外执行 `rows=65537`、N=1/2 的用例，覆盖 block 循环处理多行的同步和网格上限路径。每 dtype 实际完成 750 组合、共 1500，不将重复基准发射计为新增正确性用例。
- 所有元素与独立 FP64 稠密 Hadamard 矩阵结果比较。验收参考先经 FP32 转换，再舍入到输出 dtype，与主线已舍入参考约定一致；FP16 绝对误差严格小于 `1e-2`，BF16 严格小于 `5e-2`。同时保留相对未舍入 FP64 结果的最大误差，区分存储精度误差；不与首轮原型的误差列直接混用。
- 基线、优化和优化原位变换逐字节一致；两种方法的分步/融合 INT4 对照 CPU 对实际设备变换结果的量化，packed bytes 和 scale 必须完全一致。分配前后哨兵及输入副本用于检查边界写越界和输入被改动。
- API 契约覆盖非法 N/scale/方法、空指针、部分重叠、不对齐、大小溢出、零行及手写正负半整数舍入预期。runner 另执行 14 个 CLI 拒绝用例，要求实际返回 2。

基准默认覆盖两 dtype、N=64/128/256、rows=1/17/257/4096/16384。大形状以 batch/seq/heads 的乘积表示。每形状比较基线/优化的 transform、split、fused 六条路径，各先预热 10 次；默认 5 组，每组重复 100 次，组间轮换方法顺序。统计使用全部组样本，报告中位数、最小值、最大值以及包含退化的基线/优化比值。

计时使用同一显式 stream 的兼容设备事件，区间不含分配、主机/设备复制或 CPU 参考；输入只读重复使用，属于缓存可复用的测量。**`kernel_us` 是该发射序列的事件区间均值，不是端到端延迟，也不是隔离测得的单个内核执行时间；主机发射之间可能存在设备空闲间隔。`logical_GBs` 是逻辑张量读写量估计，不是硬件测得的物理显存带宽。**每个大基准形状另外核验量化一致性及首/中/末行稠密参考。

## 共享内存 API：MR-V100 实测结果

2026-09-05，COREX `clang++ 18.1.8 / 4.4.0` 构建实际提交 `b3267893a53e45d2e7f35dc2d6e2583c638f4112`，运行时工作树干净，五个参与构建/验证的源文件 SHA256 已与该提交的 Git 对象核对。构建日志为空、退出码 0；完整 runner 退出码 0、`quick=false`、`full_matrix=true`。环境及完整命令见 [run_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/run_summary.json)。

| 精度 | 实际变换组合 | 检查元素 | 对已舍入 FP64 参考的最大绝对误差 | 对未舍入 FP64 参考的最大绝对误差 | API 契约检查 |
|---|---:|---:|---:|---:|---:|
| FP16 | 750 | 3,318,829 | 0.00390625 | 0.015620231628418 | 61 |
| BF16 | 750 | 3,318,829 | 0.000003814697265625 | 0.124984741210938 | 61 |

两 dtype 的基线、共享内存优化版和优化版原位变换均逐字节一致，全部分步/融合 INT4 packed bytes 与 scale 对照 CPU 对实际设备输出的量化精确一致。两种误差均保留在 [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/validation.json)；**阈值判定使用已舍入参考，未舍入列包含输出存储的量化误差，不能将其也描述为低于同一阈值。**这与前文首轮原型的未舍入误差定义不同。

三次独立基准进程使用相同二进制 SHA256 `86e38be3a739548104616831a489799901d1bdba9dc9b26d825b60c030fdb93b`，均退出 0。每轮 900 条原始组样本，合计 2700 条；共有 30 个 shape/dtype 条件、六种方法和 90 组“优化方法对匹配基线”的比较。三轮分别保存为 [第一轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/benchmark.csv)、[第二轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/benchmark_run2.csv)、[第三轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/benchmark_run3.csv)，重复进程命令与退出码见 [repeat_runs.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/repeat_runs.json)。

按每轮五组的中位数比较，耗时下降定义为 `100*(1-优化耗时/基线耗时)`：

- 60/90 组比较在**每一轮**都减少至少 5% 的耗时；66/90 组每轮均更快。
- 24/90 组至少一轮退化，其中 22 组每轮均退化；4 组每轮退化超过 3%。因此共享内存优化版没有在所有输入上胜出。
- 这四组均为 N=256 的纯变换：FP16、BF16 各自的 rows=1/17。三轮退化范围分别约为 FP16 rows=1 的 4.89%～5.11%、rows=17 的 3.24%～4.28%，BF16 rows=1 的 4.76%～4.92%、rows=17 的 3.22%～4.27%。

全部样本、跨轮波动及退化比较见 [analysis.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/analysis/analysis.json) 和 [method_summary.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/analysis/method_summary.csv)。组内/跨轮标准差仅描述这几次运行的波动，不是置信区间，也不证明任意共享负载下的性能。

第二、三轮前后保存了 `ixsmi` 快照，报告 MR-V100、32GB、驱动/IX-ML 4.4.0；快照可见时钟信息，不能据此宣称独占或固定时钟。实际物理计算配额与共享状态仍未得到供应商确认。本轮不包含硬件性能计数器、端到端测量或 Warp64 结果。

### 验证脚手架初始化缺陷与修复边界

早期 quick 验证首先报哨兵被改写；随后增加阶段诊断，发现契约测试的 `initialized` 阶段、尚未执行本段算子时，输入 payload 已不符合初始化模式。原始失败分别保留在 [首次 quick 日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/prior_failures/iluvatar-api-quick-e508f98/validation.log) 和 [诊断日志](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/prior_failures/iluvatar-api-diag-e508f98/validation.log)，对应目录有完整 runner 摘要及源文件 hash。诊断版有未提交的插桩，必须按其摘要里的 `source_sha256` 和工作区状态追溯，不能只认 HEAD。

`b326789` 将脚手架缓冲区初始化改为调用方显式 stream 上的 `cudaMemsetAsync`，让初始化、上传、kernel、回读具备同一 stream 顺序，并避免初始化的异步 H2D 读取随后被上传逻辑更新的主机 shadow。同时增加初始化、非法参数、上传和量化各阶段的缓冲区检查。实际设备算法、误差阈值和测试矩阵没有通过降低要求来规避失败；修复后完整矩阵和契约检查通过。

这证明记录版本修复后通过了现有测试，也说明早期报错不能直接归因于 Hadamard 蝶形内核；**不将此记录扩展成已经证明某个 COREX 驱动缺陷或排除了所有潜在竞态。**失败运行不计入通过数量。

### 公开归档

[公开清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/shared_b326789/manifest.json) 逐文件记录字节数和 SHA256，包含原始成功日志、三轮 CSV、分析结果和前述失败证据。29 份成功运行原始文件先经传输清单校验，复制后保持字节不变；目录内 `.gitattributes` 将证据标记为 `-text`，避免 Git 换行转换破坏哈希。二进制和租赁凭据未纳入公开材料。当前工作树后续变化不改变本轮固定提交的证据范围。

首轮原型证据继续保留在前文及 [初版清单](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/manifest.json)。后续方法应新增独立记录，保留本轮历史与负例。

## Warp64：显式使用与复现

`a387db3` 在两种共享内存方法之外增加 `Method::Warp64`。每 CTA 256 个线程处理四行，一行对应完整的 64-lane warp；N≤256 时每个 lane 保存 1/2/4 个 FP32 寄存器槽。行内蝶形、最大值归约和相邻 INT4 元素交换通过经该设备验证的 64-lane shuffle 实现；尾行按整个 warp 屏蔽，网格复用时保持同一行的完整参与。仍先将变换输出舍入到 FP16/BF16，再量化，与公开 API 契约一致。

此路径要求同时满足：

1. 用真实 COREX 编译器生成天数设备代码，编译环境定义 `__ILUVATAR__`，并显式传入 `-DHADAMARD_ILUVATAR_WARP64`。不要手动伪造 `__ILUVATAR__`。
2. 实际运行设备报告 `prop.warpSize == 64`。runner 会检查；独立 API 调用方必须在初始化阶段预查，因为 API 不在每次发射时查询设备。
3. 使用与当前设备匹配的驱动/SDK，以及前文规定的连续 FP16/BF16 设备缓冲区、合法 N、正有限 scale、正确容量/对齐及 stream。

完整复现命令：

```bash
python3 platforms/iluvatar/run_platform.py --warp64 \
  --repeats 100 --groups 5 --output results/iluvatar/warp64_full_01

# --warp64 是编译选择，不是二进制运行参数。
# 已生成的该二进制再次运行基准，仍包含九条方法路径。
./results/iluvatar/warp64_full_01/validate_and_benchmark --benchmark \
  --groups 5 --repeats 100 --csv results/iluvatar/warp64_full_01/benchmark_run2.csv
./results/iluvatar/warp64_full_01/validate_and_benchmark --benchmark \
  --groups 5 --repeats 100 --csv results/iluvatar/warp64_full_01/benchmark_run3.csv
```

手动构建时，在前文 COREX 编译命令中额外加入 `-DHADAMARD_ILUVATAR_WARP64`。不启用它时，基线和共享内存方法仍可用；`Warp64` 的非空调用返回 `cudaErrorNotSupported`，不会静默切换算法。合法 `rows=0` 继续遵循无操作约定。

独立应用可在确定当前设备后调用一次以下预检查，并检查返回值；设备改变后重新检查：

```cpp
cudaError_t require_warp64_device() {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) return status;
    return prop.warpSize == 64 ? cudaSuccess : cudaErrorNotSupported;
}
```

预检查成功且已按要求编译后，在前文调用示例中显式选择 `api::Method::Warp64`。例如，仅需要量化输出时调用 `api::transform_int4(input, packed, scales, rows, 128, 1.0f, stream, api::Method::Warp64)`；变换或分步量化也显式传入同一方法。检查发射返回值，并在回读前同步相应 stream。**仅观察到 warpSize=64 不足以证明其他芯片/SDK 具备相同 shuffle 语义；其他平台仍需完整重新验证。**

## Warp64：MR-V100 完整结果

实测提交 `a387db3332c6f9b01f128dd681848260c9691281` 工作树干净，COREX 18.1.8/4.4.0 编译无输出且退出 0。完整 runner 记录 `status=PASS`、`quick=false`、`warp64_enabled=true`、`full_matrix=true`。源文件 hash 与固定提交逐一吻合；构建命令、阶段退出码和原始输出见 [run_summary.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/run_summary.json)。

| 精度 | 实际变换组合 | 检查元素 | 基线/共享版/Warp64 变换一致性 | API 契约检查 | 对已舍入 FP64 参考的最大误差 |
|---|---:|---:|---|---:|---:|
| FP16 | 752 | 4,105,264 | 全部逐字节一致 | 90 | 0.00390625 |
| BF16 | 752 | 4,105,264 | 全部逐字节一致 | 90 | 0.000003814697265625 |

这 1504 组是在共享版矩阵上加入每 dtype 两个 `rows=262145, N=1/2` 的网格复用/尾行用例，所有三种方法均对照，原位变换、输入保持、哨兵、分步/融合 INT4 的 bytes 和 scale 检查全部通过。每 dtype 新增网格用例覆盖 786,435 个元素，已包含在上表，不能重复相加。14 项 CLI 拒绝检查来自本次完整 runner，不叠加 quick 重跑。

所有元素的独立 FP64 稠密参考定义和误差阈值与共享版相同；未舍入最大误差仍分别为 FP16 `0.015620231628418`、BF16 `0.124984741210938`，不声称未舍入误差也低于已舍入阈值。实测原件见 [validation.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/validation.json) 和 [validation.log](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/validation.log)。

### 同机三轮性能

三次独立进程使用二进制 SHA256 `9496d38d25617c2eb69b0d46973e26ac2a10d59c80fba75e73767b13a5c69dac`，均退出 0。每轮 1350 条原始样本、共 4050 条：两 dtype × N=64/128/256 × rows=1/17/257/4096/16384 × 九方法 × 五组。每组 100 次重复，统计每轮的组中位数；测量边界与前文设备事件说明相同。

**Warp64 的 90/90 组匹配比较在每一轮都比基线降低至少 5% 耗时**；在本次全部形状、操作和三轮中，相对基线及共享内存优化版均未出现退化。以下范围覆盖本次全部对应形状及三轮，定义为“对照中位耗时 / Warp64 中位耗时”；不使用旧六路径实验中的耗时作分母：

| 操作 | 基线 / Warp64 | 共享内存优化版 / Warp64 |
|---|---:|---:|
| transform | 1.1907～3.3867 倍 | 1.2063～3.3927 倍 |
| split | 1.3044～5.9580 倍 | 1.2328～3.3154 倍 |
| fused | 1.5245～7.3599 倍 | 1.3253～2.9125 倍 |

最小基线降时出现在 BF16 rows=257、N=64 的 transform，三轮约为 16.015%、16.171%、16.216%。这些结论仅覆盖本次 N=64/128/256 的性能矩阵；N=1～32 虽有正确性覆盖，不能据此外推相同加速范围。

只比较 Warp64 自身的分步与融合路径，在同一轮、相同 shape/dtype/scale 的五组中位数上，`split/fused` 为 **1.2915970～1.8199551 倍**，对应融合耗时减少 **22.5765%～45.0536%**。30/30 个形状条件每轮均降低至少 5%，未观察到负收益；最小收益出现在 FP16 `[16,64,16,128]`，三轮降时约为 22.6001%、22.5765%、22.6252%。这组比值只使用当前九路径实验，不混用旧六路径的分步时间。

在相同九路径运行里，共享内存优化版自身仍有 22/90 组每轮比基线慢，其中四组每轮超过 3%；其余符合每轮至少 5% 降时的有 62 组。分析总数 152 是 **共享版 62 + Warp64 90**，不是 152 组 Warp64 收益。旧六路径实验的共享版 60 组与本轮 62 组分别保留，不能混成同一统计。

全部原始样本见 [第一轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/benchmark.csv)、[第二轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/benchmark_run2.csv)、[第三轮](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/benchmark_run3.csv)；独立进程与相同二进制的记录见 [repeat_runs.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/repeat_runs.json)。完整分析见 [analysis.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/analysis/analysis.json) 与 [method_summary.csv](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/analysis/method_summary.csv)，逐文件字节/hash 清单见 [manifest.json](https://github.com/a962695448-rgb/Learning-CUDA/blob/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results/iluvatar/warp64_a387db3/manifest.json)。目录 `-text` 规则保留原始字节；不发布租赁凭据或本机二进制。

## 本次交付范围与局限

- 已验证：MR-V100 32GB、IX-ML/驱动 4.4.0、COREX 18.1.8；连续 FP16/BF16、N=1～256 的 2 次幂、显式 stream、原位变换、INT4 分步与融合、完整 runner 和三轮同机事件基准。共享内存与 Warp64 原始结果各自固定到源码版本。
- `Method::Warp64` 通过显式 opt-in 使用；生产默认仍为 `Method::Optimized`，没有将有限性能矩阵自动扩展成所有 N 和所有设备的默认策略。
- 不提供天数 PyTorch Python 扩展；不承诺非连续张量、任意 N、越出输出类型有限范围的输入或跨设备指针。调用方负责分配容量、设备归属及 stream 生命周期。
- 未证明物理整卡独占、固定频率、跨租户干扰控制，也未取得硬件性能计数器；`ixsmi` 快照只记录对应时刻的设备状态。没有端到端复制/分配计时，不能将事件耗时当成端到端收益。
- BF16 使用存储/转换与 FP32 累加，不宣称原生 BF16 算术或矩阵单元加速。其他天数型号、其他 SDK、NVIDIA 和其他国产芯片必须分别验证。
- 后续工作按固定输入集复测、逐形状选择和证据归档推进；PR 与课程登记在项目所有者验收后单独执行。

## 后续待完成

1. 已保存并复测当前基线；后续每个优化版本继续单独保留源码、SDK、日志和哈希。
2. 若继续扩大性能矩阵或调整默认派发，为未测 N/shape 增加对照并重新回归，不宣称当前统计覆盖所有输入。
3. 当前 MR-V100 用户可在编译与设备预检查满足要求后显式使用已验证的 Warp64 路径；其他环境先保留共享实现并单独取证。
4. 补充独占/共享配额和工具权限；若报告端到端性能，另行实现并明确包含的分配/复制/同步边界。
5. 发布时核对最终用户分支提交、源码与原始证据的公开 hash。框架扩展和原生 BF16 算术支持属于额外能力，保持上述边界。

国产适配结果不能替代九齿项目要求的 A100 验证。
