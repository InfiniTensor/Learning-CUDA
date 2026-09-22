# Marlin 学习补记：从源码到 CUDA Core 实验

2026-09-22 · 0xtoward · 接续 [报告第 7 节](REPORT.md#7-marlin从独立反量化到融合-gemm)。本次补记整理 9 月 21–22 日的实验与后续计划。

## 1. 先统一问题

`C[M,N] = A[M,K] × W[N,K]^T`：A 是激活，W 是模型权重。Linear 的 M 是本次处理的 token 数；普通 decode 通常等于活跃 batch，prefill 则是调度到本轮的 token 总数。

| 路线 | 权重 / 激活 | 核心计算 | 本轮定位 |
|---|---|---|---|
| 原始 Marlin | INT4 / FP16 | Tensor Core MMA | 阅读原论文、知乎解读与源码 |
| 本地 vLLM Marlin | MXFP8 / BF16 | 片上解码 + Tensor Core MMA | 实际算子与推理对照 |
| 自写 CUDA Core 教学版 | MXFP8 / BF16 | 软件解码 + FP32 FFMA | 分离布局、复用、流水线和调度的影响 |

基础 quant/dequant 作业保持独立。这里学习的是如何让低精度权重直接参与 Linear，省去整层反量化张量的显存写回和再次读取。

## 2. 知乎九点与源码：每个 trick 在省什么

结合 [知乎解读](https://zhuanlan.zhihu.com/p/716412368)、[原始 Marlin](https://github.com/IST-DASLab/marlin) 和 [固定版本 vLLM 源码](https://github.com/vllm-project/vllm/tree/4cc0cb6f76a608622a29d9ea8f4d415697e21364/csrc/libtorch_stable/quantization/marlin) 整理：

| 优化点 | 机制 | 代价 / 适用条件 |
|---|---|---|
| 接近权重搬运下限 | 压低解码、计算、同步成本，让压缩节省的字节兑现为时间 | M 增大后可能转为计算受限；memory-bound 是分析结果 |
| 连续、向量化读取 | 离线 repack 匹配线程与 MMA 布局，减少 load、寻址和无效事务 | 权重/scale 要一起匹配；repack 是真实重排，不是 view |
| shared layout | padding 或 XOR swizzle 分散 bank 访问，同时满足矩阵指令布局 | 装入/取出映射一致，兼顾对齐与 shared 容量 |
| 多级流水线 | global→shared 的多 stage 与 shared→register 的双 fragment 缓冲分层设计 | buffer 数量不等于重叠程度；同步、shared、寄存器都有成本 |
| warp layout | 划分输出子块，或让多个 warp 分摊同一输出块的 K | 需要合并部分和；与生产者/消费者 warp 分工是两件事 |
| 快速解码 + MMA | 位操作、打包转换与矩阵计算交错，解码结果在片上消费 | INT4 与 E4M3 解码规则不同，要分别推导 |
| group scale 复用 | scale 跟随权重布局，合并读取，寄存器或 shuffle 复用 | 细粒度 scale 增加流量；长期保存在寄存器也占资源 |
| 条带式工作划分 | 把输出 tile 沿 K 切成小工作单元，减少 CTA 工作量长尾 | 增加 partial、锁或归约；小问题可能更适合 direct-grid |
| GPTQ / 校准 | 在量化前选择合适尺度、裁剪与校准样本 | 属于模型质量优化；算子 A/B 固定同一份量化权重 |

源码阅读顺序：launch 的 tile/CTA 配置 → packed layout → load/dequant → 主循环 → 输出与跨 CTA reduction。`cp.async` 的复制粒度、流水线 stage 数、寄存器 fragment 缓冲数分别看，不能合成一个概念。

## 3. CUDA Core：从 shared 走向小 M 寄存器复用

终版每个 CTA 有 4 个 warp；每个 warp 负责最多 4 行 × 1 列输出，分轮遍历完整 K。每个 lane 每轮读取 4 个连续权重，更新 4 个 acc；最后用 shuffle 合并 K 部分和。

| 调整 | 具体收益 |
|---|---|
| M tile 从 16 改为 4 | M=4 时去掉 12 行补零输出的 FFMA |
| 同一权重更新 4 行 | 摊薄权重加载和反量化 |
| Q 用 4B load，A 用 8B load | 每次分别读取 4 个 E4M3 / 4 个 BF16，减少指令和寻址 |
| 每 8 个 lane 共用 scale | 一个 lane 加载、shuffle 广播，对应 group32 |
| E8M0 用指数相加恢复 | 正常值避开通用 ldexpf；特殊范围保留 fallback |
| 正常值省 BF16 往返转换 | E4M3 有效位能被 BF16 容纳；先用独立解码逐 bit 验证 |
| 去掉 shared 中转 | 删除 shared load/store 和 CTA barrier；A 跨输出列的复用交给缓存 |

### 同输入消融，单位 µs

RTX 4070 Laptop，36 SM，SM89，WSL，nvcc 12.6，Torch 2.13.0+cu130。CUDA Event、cold L2、预热 5 次；7 轮正反交替、每轮 101 次采样并丢弃首个，取各轮中位数的中位数。量化/repack 在计时外，输入、Q/S 和输出精度固定。

| 版本 | M4,N4096,K2048 | M4,N16384,K256 |
|---|---:|---:|
| 旧 shared 双缓冲 | 181.248 | 83.968 |
| 只改 warp 内分 K，每次 1 行 | 333.824 | 179.200 |
| 同权重复用 4 行 | 148.480 | 75.776 |
| 加快速指数解码 | 139.200 | 70.656 |
| 加 4B 权重 load | 84.992 | 47.104 |
| 再加预取 | 87.040 | 48.128 |
| 再加 8B 激活 load | 69.632 | 41.984 |
| 整理后的终版 | **68.608** | **39.936** |

终版分别达到旧双缓冲的 2.64× / 2.10×。单独 warp 分 K、单独预取都出现退化；收益来自适配小 M 的工作量、数据复用和加载/解码组合。

NCU/SASS 中，同一长 K 形状的 FFMA warp 指令从 4,194,304 降到 1,048,576，原因是删除补零行；LDS/LDS.128 变为 0，终版用户 shared 为 0、45 registers/thread。L1/TEX 吞吐率下降而运行更快，说明删除了工作，不能只追求吞吐百分比更高。

## 4. 新流水线实验：有效的局部调整，不一定是更快的整体

下列两轮独立实验各为 5 轮 × 51 次、cold L2、中位数口径；形状均为 M4,N4096,K2048。不同轮次分别保留自己的基线。

| cp.async 对照 | 时间 / µs | 结论 |
|---|---:|---|
| 同布局普通复制 | 221.184 | raw shared 暂存对照 |
| cp.async 立即等待 | 226.304 | 发起后立即等，不能充分覆盖复制延迟 |
| cp.async 延后等待 | 214.016 | 比立即等待少约 5.4% 时间 |
| 同轮旧双缓冲 | 186.368 | 额外 raw→解码→float shared 和同步仍比原路径贵 |

SASS 确认为 `LDGSTS → FFMA → DEPBAR`。该版本只异步复制原始字节，解码另做；shared load wavefront 增加约 12.5%，barrier 指令数翻倍。保留为对照实验。

| Warp specialization，4 producer + 8 consumer | 时间 / µs | NCU 证据 |
|---|---:|---|
| 第一版：专职加载/解码与计算 | 968.704 | producer 供料慢，动态 barrier 编号预留 16 个槽 |
| 加生产者独立预加载 | 514.048 | 减少读取依赖，寄存器 36→56 |
| 再改常量 barrier 编号 | **277.504** | barrier 16→4；实际 occupancy 约 25%→65% |
| 同配置串行控制 | 295.936 | 流水线减少约 6.2% 时间 |
| 同轮旧双缓冲 / 无 shared 终版 | 187.392 / 72.704 | 分工和握手成本仍然更高 |

还测试了 1/2 个 producer warp。生产者和消费者使用 full/empty 命名 barrier 交接双缓冲，全部在同一个 CTA、同一条 stream 内。无需 Green Context，也未使用 Tensor Core 或 cp.async。这次认识到：寄存器、shared 之外，barrier 预留同样会影响并行度。

## 5. 精度与真实推理

| 验证 | 结果 |
|---|---|
| 小 M 终版 | 200 个 GEMM 用例通过；64,770 个有限编码/scale 组合解码逐 bit 对齐 |
| cp.async | 60 组精度、边界、非默认 stream、Graph 检查通过 |
| warp 分工 | 216 组检查通过；最大绝对误差 1.1921e-6 |
| sanitizer | 最佳异步复制/分工版 memcheck、racecheck、synccheck 通过 |

Qwen3-4B-Instruct-2507 的真实 QKV、MLP gate/up、down 权重上，M=1/4 时教学版加 BF16 输出转换的耗时为本地 vLLM Marlin 的 **1.14–1.22 倍**。对照 vLLM commit 为 `4cc0cb6f76a608622a29d9ea8f4d415697e21364`，工作树含此前实验补丁。

两分钟级 eager E2E：同一份 144 层量化 Q/S，关闭 Graph/prefix cache，每请求生成 64 tokens，每个版本每档 batch 累计至少 60 秒推理。

| batch | 教学终版 tokens/s | 本地 Marlin tokens/s | 教学终版 / Marlin |
|---:|---:|---:|---:|
| 1 | 38.33 | 46.19 | 83.0% |
| 4 | 121.16 | 175.81 | 68.9% |

本例 prefill M=22/88、decode M=1/4。所有量化 Linear 都替换为教学版，大 M 首 token 阶段差距明显；适合继续做分档与 hybrid。batch1 输出一致；batch4 四路中一路从第 21 个 token 分叉，性能测试与任务正确率分开记录。

调用边界也厘清了：QKV/O 与 MLP 是权重 Linear，走 Marlin；`softmax(QKᵀ)V` 走 FlashAttention 2，消费激活与 KV cache。本地 Marlin 接收调用者的 current stream；kernel 内部流水线不等于多个 CUDA stream。

## 6. 下一步

| 优先级 | TODO | 验收 |
|---|---|---|
| P0 | 小 M 专用与大 M 回退的 hybrid，覆盖 QKV/O/MLP | 同 Q/S、同输出 dtype，M=1/2/4/8/16/32 与 prefill 形状扫描 |
| P0 | 把寄存器/向量读取经验迁移到 Tensor Core 实现 | 从最小 MMA tile 起步，对齐 repack、scale、ldmatrix，逐项消融 |
| P1 | direct-grid、split-K、atomic 按实际工作量选择 | 精度、workspace 初始化/复用、多 stream 回归与稳定时延 |
| P1 | 精简 cp.async 的 raw/float shared 中转，重新平衡 warp 分工 | 同线程/布局控制组；同时看时延、资源和 SASS |
| P1 | NVFP4/4over6 融合 Linear，保持 tiny scale 语义 | 编码/反量化边界、真实权重 GEMM、公开选择题正确率 |
| P1 | 修复 profiler 兼容性后补全推理时间线 | 当前 NSYS/CUPTI 缺少 GPU 活动；重新确认活跃 stream 和关键路径 |
| P1 | 将稳定的 Marlin 调度/scale 修复整理为独立上游补丁 | 干净基线、多形状/类型、独占环境 A/B 与回归测试 |

测量习惯：先固定数学和数据，再改一个因素；Event 测时间，NCU 找资源与指令证据，E2E 验证系统收益。笔记本未锁频，1–2% 差异按噪声处理；NCU replay 时间不与自由运行时间互算加速比。

本补记提交学习结论和已有测量摘要；实验 kernel、完整采样和大型 profiler 报告保留在本地教学目录，不改变基础 codec 的依赖与接口。
