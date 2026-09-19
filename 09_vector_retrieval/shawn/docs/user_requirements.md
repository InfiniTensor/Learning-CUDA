# User Requirements

- 以训练营任务九为主，提供 exact 与 IVF-Flat GPU 向量检索及 CPU ground truth。
- 所有公开检索功能必须能在 CPU/GPU 上准确对照，最终覆盖任务要求的数据规模。
- 本地只要求 CPU 构建与测试；CUDA 正确性、性能和 profiler 在 SSH 服务器运行。
- 关键函数保留中文注释，明确度量、排序、索引和内存/计时边界。

### Document Preferences

- 语言：中文正文 + English 标题。
- 正确性、Recall/QPS trade-off、exact/reference 和 profiler 分节。
- 所有最终数字保留配置与原始日志。
- 正式 GPU 实验必须固定覆盖 `N=1,000,000`、`D=128`、`Q=1,000`；小规模矩阵仅作为正确性 gate，不能替代正式结果。
- 正式流程需要一条命令生成数据、构建并回读 IVF 索引、运行 exact/IVF 参数扫描、生成图表和产物索引；失败后应支持从已通过阶段继续。
- CPU exact 在百万级数据上允许使用明确记录的 query 子集估计 reference QPS，但 GPU exact 与 IVF 的正式吞吐和延迟必须基于全部 1,000 条 query。
- profiler 必须分别覆盖 exact 与 IVF，保留 ncu/nsys 原始报告、完整命令和被测配置。
