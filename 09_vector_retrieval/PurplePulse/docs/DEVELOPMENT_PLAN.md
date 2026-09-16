# 开发计划

## 第一阶段：正确性地基

- [x] 定义带版本号的二进制文件格式
- [x] 实现数据生成、读取和检查
- [x] 实现 CPU 精确检索参考答案
- [x] 实现容易理解的 CUDA 距离计算 baseline
- [x] 建立自动正确性与边界测试
- [x] 在中等规模上复验 L2、内积和 Cosine 的 CPU/GPU 一致性

## 第二阶段：精确检索性能

- [x] 将 Top-K 从 CPU 移到 GPU
- [x] 实现分块与两阶段 Top-K
- [x] 将局部 Top-K 从有序插入改为无序候选 + 最差项跟踪
- [x] 为 K=1/10 增加编译期模板特化
- [x] 实现 FP16 输入、GPU 16 位存储和 FP32 累加
- [x] 扫描 batch=1/4/8/16/32/64/128 的吞吐与延迟
- [x] 融合 Exact 距离计算与局部 Top-K，删除完整距离矩阵写回
- [x] 用 nsys 记录 Top-K 优化前后的 kernel 时间线
- [x] 完成 ncu 可用性诊断；确认赛事平台宿主权限限制并在报告中记录，不虚构计数器数据

## 第三阶段：IVF-Flat

- [x] 用确定性采样 k-means 训练聚类中心并构建倒排表
- [x] 使用连续桶布局保存和加载版本化索引
- [x] 实现 CPU 固定 nprobe 正确性基线
- [x] 对比 exact search 的 recall@K
- [x] 实现第一版 GPU 常驻索引、候选扫描和 Top-K
- [x] 将中心选择迁到 GPU，并消除候选位置 H2D
- [x] 使用每个 probe 一个 block 提高单 query 候选扫描并行度
- [x] 对相似度度量使用球面 k-means，修复大 nlist 下的桶倾斜
- [x] 使用 block 归约并行选择中心，替代单线程有序插入
- [x] 完成 100k × 128、K=10 的首轮 nlist × nprobe 扫描
- [x] 使用 warp 协作改善桶内向量读取与距离计算，并保留 scalar A/B 基线
- [x] 百万规模扫描 nlist=256/512/1024 与对应 nprobe
- [x] 在百万规模扫描 IVF batch=1/8/16/32/64/128
- [x] 对 K=1/10 完成首轮质量与吞吐对比
- [x] CLI 报告 exact/IVF 显式 GPU 缓冲区，并提供自动 CSV sweep 脚本
- [x] 使用 K=10/50/100 模板化 warp heap 扩展 IVF 局部 Top-K
- [x] 使用 probe 有序列表 k-way merge 高效归并 K=50/100
- [x] 百万规模完成 K50/K100 的 batch 与 nprobe 首轮扫描
- [x] 将 metric 编译期特化，并删除每候选无效的 warp shuffle 归约
- [x] 将小于等于 8 KiB 的 query 缓存到 block 共享内存
- [x] 为大 K 使用 compact 2-warp block，保留 warp2/4/8 A/B 路径
- [x] 优化后重新扫描 K10/K50/K100 的 batch 与 nprobe
- [x] 实现 GPU bucket-major 任务分组，验证高 nprobe 下的跨 query L2 复用
- [x] 缓存单 warp heap 根节点，以空穴下沉减少插入与排序的共享内存写入
- [x] 为单 warp probe 删除多 warp 合并同步并直接写回有序 Top-K
- [x] 评估分布式寄存器 Top-K、8 候选缓冲和 warp bitonic sort；依据
  +0.43%、-22.68% 和 -18.72% 的 A/B 结果拒绝替换稳定实现

## 第四阶段：Agent 场景创新

- [x] 根据归一化中心分数质量为每个 query 自适应选择 nprobe
- [x] 实现 masked/grouped 两种变长 GPU 调度并完成 A/B
- [x] 与相同平均 probe 的固定 nprobe 完成百万级公平实验
- [x] 根据负结果保留实验接口但不替换固定策略默认值
- [x] 在 GPU 候选扫描中融合元数据过滤和记忆评分
- [x] 与分离重排序方案进行公平实验

## 第五阶段：正式实验与报告

- [x] 百万向量、128 维、1000 queries 固定数据集
- [x] 记录 QPS、run/batch P50/P99、显存和 recall@K
- [x] 完成 FAISS CPU/GPU Flat/IVF-Flat 与自研引擎首轮公平对照
- [x] 补齐 K=50、batch=40 的 FAISS GPU 同批次 100 次正式对照
- [x] 保存 FAISS 与自研 nprobe 扫描原始 CSV
- [x] 输出 recall-latency、recall-QPS 和 batch-QPS 曲线图
- [x] 记录赛事平台无法开放 ncu 计数器的原因、影响范围和替代证据
- [x] 完成最终总结报告与复现说明

Synthetic-1M 最终结果中，WarpFusion K10 在 nprobe=160 下比同口径 FAISS
GPU IVF 快 1.20 倍。K50 经 batch=40、bucket-major 和 4×16 分区局部 Top-K
优化后达到 7934 QPS，与 FAISS 的差距缩小到 4.97%；K100 使用 4×40 分区
局部 Top-K 达到 7205 QPS，差距缩小到 9.64%。Exact fused 在百万规模
K10/K50/K100 上相对矩阵路径分别加速约 3.13×/7.16×/10.59×，显式 GPU
缓冲区减少约 33%，但仍比 FAISS GPU Flat 慢 3.18--3.71 倍。上述结果已经
通过正式 A/B、输出一致性检查、CTest 和 Compute Sanitizer 验证。

## 第六阶段：真实公开数据验证（已完成）

目标是验证 WarpFusion 的优化能否从 Synthetic-1M 推广到真实向量分布。
继续使用当前 RTX 4090，不以更换硬件作为达标条件。

- [x] 接入 SIFT1M，并保留公开数据校验和、转换命令和 ground truth。
- [x] 接入至少一套 384 或 768 维公开文本 embedding 数据。
- [x] 在两套真实数据上完整评测 K=10/50/100 与
  nprobe=32/64/128/160/192/224。
- [x] 同时报告 WarpFusion、FAISS GPU IVF-Flat 和 Exact ground truth 的
  QPS、P50/P99、recall、显存与建库时间。
- [x] 验证分区局部 Top-K 的外部有效性；若产生不可接受的 recall 损失，提供
  自动回退到完整 heap 的安全路径。

验收门槛：真实数据上输出正确，实验可以一条命令复现，并形成至少两张完整的
recall-QPS 曲线。WarpFusion 在相近 recall 下保持接近 FAISS 的竞争力，而非
只在 Synthetic-1M 上成立。

验收结果（2026-09-10）：通过。SIFT1M 与 384 维 Yahoo MiniLM 的完整矩阵、
公开 ground truth 交叉验证、FAISS GPU 对照和四张曲线均已落盘。在
`nprobe=160` 下，WarpFusion 在六组 K/数据集组合中达到 FAISS GPU IVF-Flat
的 1.33--2.53 倍吞吐。K=50 分区路径在两套真实数据上触发 recall 安全回退，
K=100 分区路径保持零 recall 损失并分别加速 1.38 倍和 1.72 倍，证明自动回退
是必要的，而不是只保留未验证的快速路径。

## 第七阶段：Exact fused 性能冲刺（已完成）

- [x] 分离距离计算、局部选择和最终归并耗时，确定 K=10/50/100 的真实瓶颈。
  2026-09-10 的 differential profile 显示：Synthetic-1M 的距离阶段占
  80.72%--98.26%，Yahoo MiniLM 384 占 89.58%--98.73%；最终归并最多
  1.03%，不是当前主瓶颈。局部 Top-K 使用 `fused - distance-only` 估算，
  不是硬件计数器的独立时间线。
- [x] 比较当前 heap、分层 Warp/Block Top-K、候选缓冲后归并以及块级
  radix/select 方案。当前 fused 本身采用 8 个 warp 局部 heap + block 归并；
  已实测的 matrix K=10 候选缓冲后归并、K=50/100 block 反复选择路径慢
  3.13--10.59 倍且多用约 33% buffer。
  全量 radix/select 若保持精确语义，只能物化 score 或重复距离扫描，违反本阶段
  fused 约束，故在设计审查后不实现；不将其写成 GPU 实测结果。
- [x] 对不同 K 独立选择实现，不要求一套 kernel 覆盖所有场景。首轮
  query-tiling 结果选择 K=10/128 维 tile4、K=10/384 维 tile8、K=50/384 维
  tile4；K=50/128 维和 K=100 回退原 fused heap。
- [x] 保持融合路径不写回完整距离矩阵，且候选 ID 与精确参考逐项一致。
  query tile 的全部 smoke 与正式 A/B 均通过逐 query、逐 rank ID/score 校验，
  Compute Sanitizer 报告 0 errors。

验收门槛：在 Synthetic-1M 和至少一套真实数据上，WarpFusion Exact 相对
FAISS GPU Flat 的吞吐差距缩小到 2 倍以内；若未达到，必须用可复现 profile
明确剩余瓶颈，并记录被否决方案和原因。

阶段进展（2026-09-10）：跨 query 复用数据库读取的 tiled fused kernel 已完成
实现和三数据集正式验证。Synthetic-1M/SIFT1M K=10 分别提升 37.3%/33.0%；
Yahoo K=10 使用 tile8/160 blocks 提升 88.7%，K=50 使用 tile4 提升 21.1%。
128 维 K=50 和全部 K=100 路径退化，已记录并回退。优化后相对 FAISS GPU
Flat 的 K=10 差距仍为 Synthetic 2.30 倍、SIFT1M 2.24 倍、Yahoo 3.15 倍，
尚未达到本阶段 2 倍门槛。后续 block 级数据库 tile 在 Synthetic/Yahoo 的
最佳值分别为 9050/4794 QPS，均低于已接纳的 warp-query 9265/6989 QPS，
因此记录为负实验且未扩展到正式长跑。K 专用选择结构的最终裁决见上方清单与
复现说明：保留当前分层 heap，不引入需要完整 score 物化或重复扫描的方案。

验收结论（2026-09-10）：按计划的失败分支完成。吞吐差距未缩小到 2 倍以内，
但已有可复现 differential profile 明确显示距离/数据库读取占 80.72%--98.73%，
且 cosine 范数预计算与 block-tile 两条候选均保存负结果和原因。选择结构比较
确认完整候选物化、重复距离扫描和全量 radix/select 不能在保持 fused 约束下
解决主瓶颈；已接纳的 query-tile 逐 K 回退，默认行为不扩大到退化组合。

## 第八阶段：稳定性与适用范围（已完成）

- [x] 覆盖维度 64/128/384/768，batch 1/8/32/40/64/128，以及
  K=1/10/50/100。
- [x] 覆盖 FP32/FP16、L2/inner-product/cosine、空桶、倾斜桶、重复向量、
  相同距离和非整 warp 维度。
- [x] 所有默认模式通过 CTest、Python 回归测试和 Compute Sanitizer。
- [x] 给出按 K、维度和 GPU 资源选择实现的规则，并为非推荐组合提供安全回退。

验收结果（2026-09-10）：通过。新增独立 `stability_tests`，完成 576 组 Exact
交叉矩阵、6 组满 batch + 尾 batch、6 组 65 维 dtype/metric 组合、3 组已接纳
query-tile 和 2 组 fail-closed 回退。另构造 8 桶中 4 个空桶、最大桶占
125/128 的病理 IVF 索引，完成 48 组 warp/bucket-major CPU/GPU 对照。
远端 CTest 14/14 通过，稳定性矩阵 Compute Sanitizer 报告 0 errors；选择规则
和边界记录在 `docs/STABILITY_MATRIX.md`，原始摘要位于
`results/stage8_stability/coverage.csv`。

## 第九阶段：工程化与报告定稿（已完成）

- [x] 提供稳定的构建、建库、查询和 benchmark CLI，减少与本机目录绑定的配置。
- [x] 提供一条命令完成数据准备、测试、FAISS 对照、绘图和报告数据校验。
- [x] 将真实数据结果、Exact 新实现、负实验和适用边界写入技术报告。
- [x] 补充 API 示例、默认配置表、架构图、性能表和已知限制。
- [x] 发布前在一套干净 Linux + CUDA 环境完成从零构建和最小查询验证。

验收门槛：四个技术阶段的代码、实验、原始 CSV、图表、报告和复现说明相互
对应；课程提交版本与后续开源候选共用同一套可追溯证据。

验收结果（2026-09-11）：通过。新增项目根目录无关的 `scripts/reproduce.py`，
串联构建、18 项 CTest、两套公开数据准备、WarpFusion/FAISS 实验、绘图、PDF
和发布校验，并以原子 checkpoint 支持 `--resume`。`scripts/validate_release.py`
校验正式 CSV、Stage 8 覆盖、SVG、核心文档和 PDF；`docs/API.md` 与 README
补齐 CLI/C++ 示例、默认配置、代表性能和限制。RTX 4090 D 的 CUDA 12.8 干净
构建通过 18/18 CTest，257×65、13 queries 的 Exact/IVF CPU/GPU 最小查询均
通过候选与分数校验；摘要证据位于 `evidence/2026-09-11/stage9/`。

## 第十阶段：开源发布冲刺（条件阶段）

只有第六至第九阶段全部通过验收后，才启动本阶段。目标是把完成验证的工程
整理为可公开发布、可独立复现的 GPU 向量检索研究原型，而不是先发布再补证据。

- [ ] 完成面向外部使用者的 README、快速开始、贡献指南和变更记录。
- [ ] 清理提交历史中的凭据、机器地址、临时文件和不可公开数据。
- [ ] 选择并加入开源许可证，列清第三方依赖及其许可证。
- [ ] 发布版本必须绑定源码提交、配置、数据校验和和原始 CSV，确保报告数字可追溯。
- [ ] 建立公开 issue 模板、版本标签和最小持续集成检查。

完成第六至第十阶段后，才将仓库标记为 WarpFusion 的首个公开版本候选。
建议版本号为 `v0.1.0`，定位为“单卡、内存内、Exact 与 IVF-Flat GPU 向量
检索研究原型”，不宣称生产级服务能力。

## 第十一阶段：Agent 应用方向（暂停，不纳入当前提交）

> 本节以下内容属于 Agent 应用方向，不改变项目主体。本项目仍是一套
> 可独立构建、运行和评测的 C++/CUDA GPU 向量检索引擎；Agent 是用于验证
> 元数据融合检索价值的真实工作负载，Pi 只是首个轻量客户端。

### 11.1 范围与边界

- GPU 检索核心继续负责 Exact、IVF-Flat、GPU Top-K、常驻索引、批量
  查询、元数据前置过滤以及语义/时间/重要性融合评分。
- Agent 侧只负责提取记忆、生成 query、调用检索接口和组装上下文，不在
  检索引擎内嵌 LLM，也不让 Agent 框架接管索引与排序逻辑。
- 第一版采用 session 结束后的批量摄取和下一 session 的在线检索；暂不实现
  实时增量索引、分布式服务、多 Agent 编排或 Web UI。
- 接口保持框架无关，Pi adapter 放在 `clients/pi/`，后续可以增加其他 Agent
  客户端而无需修改 CUDA 核心。

### 11.2 最小可交付链路

- [ ] 定义 Agent Memory JSONL 格式，包含 content、memory_type、project_id、
  session_id、timestamp、importance 和 source_type。
- [ ] 实现 session ingestion，将事实、经历、成功步骤和失败教训映射到文本、
  embedding 与现有项目 metadata 文件。
- [ ] 增加 embedding provider 抽象；演示实现可替换，不把具体模型写死在
  检索引擎中。
- [ ] 保存 `vector_id -> content/provenance` 映射，使 GPU Top-K ID 能恢复成
  可注入 Agent 上下文的记忆文本。
- [ ] 提供最小常驻查询接口，复用已加载的 IVF 索引和 GPU 工作区，避免每个
  Agent tool call 都重新初始化 GPU。
- [ ] 实现框架无关的 `memory_search` 与 `memory_record` 客户端协议。
- [ ] 实现 Pi adapter：`memory_search` 工具、session 结束记忆记录和带来源的
  context builder。
- [ ] 建立双 session CUDA 开发演示：后一 session 在空上下文下召回前一
  session 的成功方案或失败教训。

### 11.3 评测与验收

- [ ] 对比 No Memory、纯语义检索和本项目 fused memory 三组方案。
- [ ] 记录记忆 Recall@K/MRR、错误重复率、任务成功率和注入 token 数。
- [ ] 记录检索 QPS、P50/P99、GPU kernel 时间、显存以及错误 project/session
  的过滤泄漏率。
- [ ] 验证 Agent adapter 关闭后，原有 CTest、百万规模 benchmark 和 CLI
  行为完全不变。
- [ ] 在 README 中先展示 GPU 引擎架构与性能，再将 Agent Memory 作为应用
  案例；Agent 内容不覆盖或替代 Exact/IVF/FAISS 主线。

### 11.4 本阶段明确不做

- MiniDeepSeek 或其他小模型训练。
- MoE、GRPO、DPO 或 learned retrieval policy。
- Base/Delta 在线索引、后台 compaction 和分布式多租户调度。
- 完整 RAG、复杂网页、多个 Agent 框架同时适配。

完成上述最小链路后再根据时间评估在线增量索引；任何扩展不得降低
GPU 检索核心测试、性能复现和代码可解释性。
