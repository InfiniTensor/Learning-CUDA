# Idea Report

## Part 1 — Motivation

任务九需要先建立可证明正确的 GPU exact baseline，再以 CPU exact 为 ground truth 评估 IVF-Flat 的 Recall、延迟、吞吐和显存占用。

### Research Questions

- RQ1：CUDA exact 在 L2/IP/Cosine、FP16/FP32 和不同 K 下能否与 CPU 排序一致？
- RQ2：`nprobe` 与 batch size 如何形成 Recall/QPS/P99 trade-off？
- RQ3：全量 radix sort 的瓶颈是否支持后续 hierarchical Top-K 优化？
- RQ4：逐 query GPU 路径下，batch size 是否实际改变性能，还是仅形成调度分组？

### Key Works

| 方法 | 工程用途 |
|---|---|
| IVF-Flat | coarse partition 与候选精排基线 |
| CUB radix sort | CUDA exact 全排序正确性基线 |
| FAISS | 后续拟补充的同数据、同计时口径外部参考；本次正式实验未运行 |

## Part 2 — Method

CPU 使用固定大小 heap 和 score/id 确定性排序；CUDA 先物化 score/id，再用 CUB 排序。IVF 训练、持久化、coarse probe 和候选 rerank 共享 CPU/CUDA assignment 接口。

CUDA radix sort 对相同 score 保持输入顺序，因此 IVF 候选在送入 CPU/CUDA rerank 前统一按 vector ID 排序，使稳定排序自然实现“score 优先、较小 ID 优先”的全局 tie policy。服务器 gate 除分别对照 CPU exact 的 Recall/排序/score error 外，还直接检查 CPU/GPU 的 Top-K ID 集合、按 vector ID 对齐后的 score error 和容差感知排序。严格逐 rank 一致率继续作为诊断指标；仅当发生换位的两个 rank score 落在数值容差内时，才把它视为浮点等价顺序，避免把 CPU 与 CUDA 的微小舍入差异误判成检索错误。

## Part 3 — Experiment Design

正确性覆盖三种度量、两种输入 dtype、K=1/10/50/100、cosine 零向量、重复分数、索引回读和 IVF `nprobe=nlist`。服务器矩阵对 exact 与 IVF 分别执行 CPU/GPU gate；硬门禁要求 Top-K ID 集合完全一致、按 ID 对齐的最大 score error 不超过容差且容差感知 rank agreement 为 1。完全相等 score 的 vector-ID tie policy 仍由专门的 CUDA CTest 严格约束。正式规模至少 N=1,000,000、D=128、Q=1,000。

正式实验使用固定 seed 的 clustered synthetic FP32/L2 数据。GPU exact 与 IVF 对全部 1,000 条 query 计时；CPU exact 对前 20 条 query 计算 ground truth 与 reference QPS。Exact 扫描 K=1/10/50/100；IVF 分别只改变 `nprobe` 和 batch size，保留 warm-up、五次重复、原始日志和 SHA-256。若 batch 曲线近似不变，将其解释为当前 CUDA baseline 仍逐 query 执行，而不是宣称批处理优化已经完成。

## References

- Johnson, Douze and Jégou (2017), *Billion-scale similarity search with GPUs*.
- NVIDIA CUB, *DeviceRadixSort* documentation.
