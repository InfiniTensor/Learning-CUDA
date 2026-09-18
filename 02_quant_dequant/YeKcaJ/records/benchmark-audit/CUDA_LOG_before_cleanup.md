# 优化日志

## 第 0 次：优化前基准（2026-09-10）

条件：RTX 3060 Laptop，FP32 输入，block / nearest，预热 3 次、计时 20 次。
以下为设备驻留耗时，单位 ms；反量化列为中位数。

| 格式 | 元素数 | 量化中位数 | 量化 P95 | FP32反量化 | FP16反量化 | BF16反量化 |
|---|---:|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.123904 | 0.130048 | 0.031744 | 0.024576 | 0.027648 |
| MXFP8 | 4M | 0.467968 | 0.474112 | 0.111616 | 0.103424 | 0.108944 |
| MXFP8 | 16M | 1.634304 | 1.679360 | 0.404928 | 0.364544 | 0.381952 |
| NVFP4 | 1M | 0.142336 | 0.178048 | 0.025600 | 0.020272 | 0.022528 |
| NVFP4 | 4M | 0.450944 | 0.470016 | 0.088064 | 0.078848 | 0.086032 |
| NVFP4 | 16M | 1.742800 | 2.141184 | 0.339824 | 0.302080 | 0.332800 |

nsys 定位（4M，本次采集的 kernel 中位数）：
- MXFP8：编码 0.425761 ms，scale 计算 0.102718 ms，优先优化编码。
- NVFP4：scale 计算 0.408688 ms，编码 0.091735 ms，全局归约两阶段合计约 0.072145 ms，优先关注 scale。

正确性：7/7 核心测试通过。尚未修改 kernel。
数据来源：[benchmark](../../records/01-before/benchmark/RESULTS.md)、[MXFP8 nsys](../../records/_archive/duplicate-runs/opt02-v1-profile/mxfp8_stats.txt)、[NVFP4 nsys](../../records/_archive/duplicate-runs/opt02-v1-profile/nvfp4_stats.txt)。

## 第 1 次：MXFP8 直接编码 + scale 融合（2026-09-10）

改动：nearest 用 FP32 指数/尾数直接生成 E4M3；block 模式归约、scale 广播、编码合为一个 kernel。随机舍入、NVFP4 和反量化算法不变。

量化性能（同第0次条件，单位 ms；加速比相对第0次中位数）：

| 元素数 | 优化前 | 仅直接编码 | 直接编码+融合 | 融合 P95 | 加速比 |
|---|---:|---:|---:|---:|---:|
| 1M | 0.123904 | 0.055728 | 0.034816 | 0.040960 | 3.56x |
| 4M | 0.467968 | 0.199680 | 0.123312 | 0.128896 | 3.79x |
| 16M | 1.634304 | 0.708608 | 0.435648 | 0.737280 | 3.75x |

nsys：4M 默认路径由两个 kernel 合为 `mxfp8_quantize_fused_kernel`，中位数 0.143548 ms（与 benchmark 分开计量）。
复测：1M/4M/16M 中位数为 0.034816/0.122880/0.432128 ms；16M P95仍有波动。NVFP4 16M同环境旧/新对照为1.746352/1.737728 ms，未见稳定回退。
正确性：7/7测试通过，覆盖中点两侧、随机FP32位模式和跨scale分组；Compute Sanitizer memcheck/racecheck/synccheck均0错误。
结论：保留；本轮仅提升量化，反量化未优化。
数据：[分步](../../records/02-mxfp8-direct-encode/benchmark/RESULTS.md)、[融合](../../records/03-mxfp8-fused-scale/benchmark/RESULTS.md)、[复测](../../records/03-mxfp8-fused-scale/repeat/RESULTS.md)、[nsys](../../records/03-mxfp8-fused-scale/profile/mxfp8_stats.txt)。

## 第 2 次：NVFP4 block scale 直接编码 + 归约融合（2026-09-11）

定位：nsys 显示 NVFP4 默认路径中 `build_block_scales` 占 71%（0.409 ms），
编码 `quantize_kernel` 仅 16%（0.092 ms）——瓶颈与 MXFP8 相反。该 kernel 读
16 MB 按带宽约需 53 us，实测 409 us，慢 8.5 倍，故为计算受限：NVFP4 的
block scale 也是 E4M3，却仍走 7 轮二分查找（每轮 `decode_e4m3` 内含 `scalbnf`）。

改动（两步，分别计量以便归因）：
1. **scale 直接编码**：`scale_code()` 的 NVFP4 分支改调用第 1 次已有的
   `encode_e4m3_nearest()`（由 `encode_mxfp8_nearest` 改名，因两种格式共用）。
   该函数与二分路径在中点判定上同为"严格大于中点才进位"，舍入规则不变。
2. **归约与编码融合**：新增 `nvfp4_quantize_fused_kernel`，一个 warp 覆盖 4 个
   分组，组内用 `__shfl_down_sync(width=kBlockSize/2=8)` 归约最大值（每 lane
   打包 2 元素，故 8 lane = 16 元素 = 1 组），组首算出 scale 后广播，复用寄存器
   中的原值编码并打包。`input` 由读 3 遍降为读 2 遍。
   `maximum`/`finalize_max` 必须保留：`global_scale` 依赖全张量最大值，
   是固有的两阶段全局归约，无法并入 per-block kernel。

量化性能（同第 0 次条件，单位 ms，`resident_gpu` 中位数）：

| 元素数 | 优化前 | 步骤1 后 | 步骤2 后 | P95 | 加速比 |
|---|---:|---:|---:|---:|---:|
| 1M | 0.142336 | 未单独测量 | **0.063488** | 0.064512 | 2.24x |
| 4M | 0.450944 | 0.238592 | **0.142336** | 0.143360 | 3.17x |
| 16M | 1.742800 | 未单独测量 | **0.519136** | 0.523264 | 3.36x |

步骤 1 的归因仅在 4M 做了单独测量（单次运行）：0.520704 -> 0.238592 ms，2.18x。
1M/16M 未在步骤 1 状态下单独采集，故不填数字；表中"步骤2 后"为
`04-nvfp4-e4m3-scale-fused/benchmark/` 的正式结果。

nsys 验证（4M，`--benchmark` 含预热与 baseline 对照，故实例数多于单次量化）：

| kernel | 实例数 | 中位数 (ms) | 归属 |
|---|---:|---:|---|
| `build_scales` | 9 | 0.719 | baseline / tensor 路径 |
| `maximum` | 26 | 0.062 | 所有路径的全局归约第一阶段 |
| `finalize_max` | 26 | 0.003 | 同上第二阶段 |
| `nvfp4_quantize_fused_kernel` | 17 | 0.094 | **默认 NVFP4 路径（新）** |
| `quantize_kernel<(bool)1>` | 9 | 0.090 | baseline 枚举路径 |
| `dequantize_kernel<*>` | 8 | 0.09-0.10 | 三种输出类型 |

关键点：
- **`build_block_scales` 与 `quantize_kernel<(bool)0>` 在本次采集里完全消失**
  （各 0 次），默认路径已由单个 `nvfp4_quantize_fused_kernel` 取代，
  其实例数 17 与默认路径调用次数一致。
- NVFP4 的 baseline 对照路径用的是 `build_scales`（9 次），**不是**
  `build_block_scales`；后者仅服务 MXFP8 stochastic 与 MXFP8 baseline。
- `maximum`/`finalize_max` 各 26 次且成对出现，说明全局归约仍是每个量化
  调用一次，未被融合消除（`global_scale` 依赖全张量最大值，属固有前置）。

正确性：
- ctest 7/7 通过（含 `--self-test` 与 CPU oracle 逐字节比对）。
- `--self-test`：`pipeline_cases=299 PASS`，`encoder_samples=10046` 且随机舍入
  比例 0.4936（合格区间内）；三种输出类型 `element_mismatches=0`、`max_abs_diff=0`。
- **专项分组边界测试 15/15 通过**：针对融合 kernel 的 `width=8` 分组，构造组基准
  逐组放大且组内递增的输入，覆盖 16/17/31/32/33/48/255/256/257/512/1000/4096/
  8191/8192/8199 等规模，由内部 `compare_packed` 与 CPU oracle 全字段比对。
  若 shuffle width 或组首判断写错导致跨组串用 scale，此测试必然失败。
- **边界值测试 16/16 通过**（两种格式各一次）：在 `--self-test` 已覆盖的
  冻结样例（basic/outlier_tail/random/tail_block/zeros，共 5 份，文件均完整）
  之外，另用 `edge_value_test.py` 针对本次改动的 scale 编码器补做边界验证：
  全零、负零、奇数长度全零、±FLT_MAX、最小正 subnormal、次正规混合、
  ±1e-38、±1e38、每组仅首/末元素有效、跨组幅值递增/递减、零与最大值交替，
  共 16 例，均由内部 `compare_packed` 与 CPU oracle 逐字节比对。
- 未回归 MXFP8：复用 `scale_code()` 使两条路径共享代码，实测 1M 0.034816、
  4M 0.123632、16M 复测 0.435664/0.434096/0.435664 ms，与第 1 次的
  0.034816/0.123312/0.435648 一致（记录中 16M 曾出现 0.477184，复测 4 次
  确认为冷启动噪声，非回退）。
- 反量化未改动：4M 为 fp32 0.088064、fp16 0.079200、bf16 0.086896 ms。

**关于 baseline 参照的说明（重要）**：`scale_code()` 被默认路径、baseline
对照路径和 tensor 模式共用，因此本轮的 scale 改动让 `quant_enumeration`（baseline）
也一并变快，实测（`resident_gpu` 中位数，ms）：

| 元素数 | `01-before` | 本轮后 | 
|---|---:|---:|
| 1M | 0.316416 | 0.210944 |
| 4M | 1.144832 | 0.726528 |
| 16M | 4.514304 | 2.827792 |

这意味着：

- `quant_enumeration` **不再是纯粹的"优化前"参照**——它只保留枚举式元素编码，
  scale 计算已是优化后的实现。
- 因此本轮的加速比一律以 `records/01-before` 为基准，而非同批次的
  `quant_enumeration`。`01-before/benchmark/environment.json` 中的 SHA256 可佐证
  它采集于任何改动之前。
- 两条路径的输出仍与 CPU oracle 逐字节一致（`compare_packed` 在每次
  `--benchmark` 内部校验），故 baseline 作为正确性对照依然有效。

**验证边界（重要）**：本轮**未能**运行 Compute Sanitizer。Deb 包
`nvidia-cuda-toolkit` 自带的是 2022.4.1 版，与当前驱动 596.08 不兼容：
默认报 `Unable to find injection library libsanitizer-collection.so`，
加 `--injection-path /usr/lib/nvidia-cuda-toolkit/compute-sanitizer` 后报
`Target application terminated before first instrumented API call`。
用最小 CUDA 程序验证，失败可复现，确认为环境问题而非本项目代码问题。
第 1 次的 memcheck/racecheck/synccheck 记录（`0 errors`/`0 hazards`）产生于
2026-09-10 环境仍可用时，其证据保留在 `03-mxfp8-fused-scale/validation/`。
本轮以 CPU oracle 逐字节比对 + 分组边界专项测试作为替代证据。

结论：保留；NVFP4 量化提升 2.24x/3.17x/3.36x，MXFP8 未回退。
数据：[benchmark](../../records/04-nvfp4-e4m3-scale-fused/benchmark/RESULTS.md)。

## 第 3 次：NVFP4 全局归约减少同步（2026-09-14）

改动：`Core/kernels/reduce.cuh` 的两个归约 kernel 改用 warp 归约，再合并 8 个 warp 结果；整块同步由 9 次减为 1 次，共享内存由 256 个 float 减为 8 个。读取方式、global_scale 公式、编码与舍入不变。MXFP8 默认 block 路径不使用这两个 kernel，tensor 路径共用改动。

条件：RTX 3060 Laptop，FP32，block/nearest，预热 3 次、计时 20 次；NVFP4 完整驻留量化序列，单位 ms。

| 元素数 | 本轮前 median | 本轮后 median | 本轮后 P95 | 本轮加速比 | 相对第一轮加速比 | 相对第 0 次累计加速比 |
|---|---:|---:|---:|---:|---:|---:|
| 1M | 0.063488 | 0.048128 | 0.049056 | 1.32x | 1.32x | 2.96x |
| 4M | 0.145856 | 0.140288 | 0.148480 | 1.04x | 1.01x | 3.21x |
| 16M | 0.520064 | 0.518144 | 0.534528 | 1.00x | 1.00x | 3.36x |

加速比 = 对应基准中位数 / 本轮后中位数。“第一轮”指 NVFP4 首轮优化完成后的结果（本日志第 2 次“步骤2 后”：0.063488 / 0.142336 / 0.519136 ms）；第 0 次基准为 0.142336 / 0.450944 / 1.742800 ms。本轮前为重新实测，因此与第一轮记录略有差异；相对第一轮和第 0 次均为跨次测量对比。

nsys（4M）：`maximum` 中位数 0.062131 → 0.056339 ms，`finalize_max` 0.003456 → 0.003217 ms。
交换顺序复测：1M 减少 25.9%，4M 减少 3.2%；16M 反而增加约 0.6%，后 P95 为 0.795648 ms，因此不声称 16M 有收益或尾延迟改善。
正确性：8/8 核心测试通过；新增 220 组归约边界测试；归约专项及两种格式自测的 memcheck/racecheck/synccheck 全部 0 错误，memcheck 无泄漏。
结论：保留，收益主要在 NVFP4 小中规模量化；MXFP8 默认量化与反量化算子不变。
数据：[优化前](../../records/06-warp-reduction/before/benchmark/RESULTS.md)、[优化后](../../records/06-warp-reduction/after/benchmark/RESULTS.md)、[复测](../../records/06-warp-reduction/repeat/summary.json)、[验证](../../records/06-warp-reduction/VALIDATION.md)。

## 第 4 次：MXFP8 向量化加载与打包写回（2026-09-14）

改动：新增 `mxfp8_quantize_vectorized_kernel`。每个线程读取 4 个连续 FP32，使用 `float4` 加载并将 4 个 E4M3 结果合并为一次 32-bit 写入；8 个线程共同处理一个 32 元素 scale 分组。尾部不足 4 个元素时回退标量读取。scale 计算、舍入和输出格式不变。

条件：RTX 3060 Laptop，FP32，block/nearest，预热 3 次、计时 20 次；单位 ms，`resident_gpu` 中位数。

| 元素数 | 第 1 次后 | 本轮后 | 本轮 P95 | 相对第 1 次加速比 |
|---:|---:|---:|---:|---:|
| 1M | 0.034816 | 0.025600 | 0.094080 | 1.36x |
| 4M | 0.123312 | 0.070656 | 0.073728 | 1.74x |
| 16M | 0.435648 | 0.269824 | 0.279552 | 1.61x |

正确性：MXFP8 回归、冻结哈希、IO 测试均通过；`--self-test` 的 299 个 pipeline cases 通过。NVFP4 路径未改动，仅作为同批次运行对照。

结论：当前中位数显示向量化版本有效，4M/16M 收益较稳定；1M 的 P95 受启动噪声影响，需后续 nsys/重复采样确认尾延迟。数据：[benchmark](../../records/07-mxfp8-vectorized/RESULTS.md)。

## 第 5 次实验：workspace 复用对照（2026-09-14）

新增 `quant_reused_workspace / host_api`：复用现有 Workspace 的显存和 event，仍计入上传、量化、下载及主机结果分配/释放。与每次新建的 `quant_optimized / host_api` 交替测量；各预热 3 次、计时 20 次，FP32 block/nearest。此轮仅新增基准对照，kernel 和单次文件 CLI 未改变。

| 格式 | 元素数 | 每次新建 median ms | 复用 median ms | 加速比 |
|---|---:|---:|---:|---:|
| MXFP8 | 1M | 2.171398 | 1.037373 | 2.09x |
| MXFP8 | 4M | 4.777892 | 2.792794 | 1.71x |
| MXFP8 | 16M | 16.660836 | 11.450560 | 1.46x |
| NVFP4 | 1M | 1.746878 | 0.963798 | 1.81x |
| NVFP4 | 4M | 4.598268 | 2.726448 | 1.69x |
| NVFP4 | 16M | 14.593807 | 9.878531 | 1.48x |

复测：4M 的 MXFP8/NVFP4 分别为 1.71x/1.70x，16M 均约 1.48x；1M 收益存在但幅度有波动。各规模两轮 P95 均下降。8/8 核心测试通过，两轮基准均通过复用结果的完整 packed 比较。

结论：复用对同进程重复调用有效，不代表单次 CLI 或 kernel 本身获得相同加速；首次创建和最终释放不计入复用耗时。正式批量入口尚未实现。
数据：[首轮](../../records/10-workspace-reuse/benchmark/RESULTS.md)、[复测](../../records/10-workspace-reuse/repeat/RESULTS.md)。

## 第 6 次：NVFP4 全局归约 partial 数量调优（2026-09-14）

改动：将 `maximum`/`finalize_max` 使用的 partial block 上限从 4096 降为 1024。每个线程通过原有 grid-stride loop 扫描更多输入，partial 的分段规则和最大值结果保持不变；没有改变 scale 公式或编码。

| 元素数 | 原实现 median ms | partial=1024 median ms | 加速比 |
|---:|---:|---:|---:|
| 1M | 0.048928 | 0.041344 | 1.18x |
| 4M | 0.143360 | 0.140224 | 1.02x |
| 16M | 0.522144 | 0.518656 | 1.01x |

复测结果：1M 为 0.041344 ms，4M 为 0.140224 ms，16M 为 0.518656 ms。Nsight Compute 的 4M `maximum` kernel 为 60.83 us，原实现为 61.31 us，说明主要收益来自减少小规模 kernel 调度/归约开销，大规模访存阶段仍接近带宽上限。

正确性：8/8 核心测试通过，归约边界测试通过；曾尝试直接将 maximum 改成连续 `float4` 加载，但会改变 partial 分段并被 `n=257` 专项测试捕获，已撤销该方案。

结论：保留 partial=1024；收益主要在 1M，小规模和大规模均无明显回退。NVFP4 融合量化的计算受限部分尚未改变。
数据：[首轮](../../records/11-nvfp4-partials1024/RESULTS.md)、[复测](../../records/11-nvfp4-partials1024/repeat/RESULTS.md)、[NCU](../../results/ncu/nvfp4-maximum-1024.ncu-rep)。
