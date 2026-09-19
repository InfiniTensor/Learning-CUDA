# 报告证据与提交对照

本文件保存从 PDF 中移出的版本、证据路径和提交管理信息，不属于报告正文。
报告源文件为 `../REPORT.md`；构建记录位于仓库 `output/pdf/report_build_manifest.json`。

## 实验来源

- 正式 GPU 实验提交：`b3e629ab3f24735bd7d16b6583dd21ea4da6bcec`，运行时工作树 clean。
- 本地完整归档：`output_task/task-09-gpu-vector-search/outputs/task9_report_b3e629a/`（相对仓库根目录）。
- 随报告提供的轻量证据：`../report_assets/evidence/`。
- 证据清单及 SHA-256：`../report_assets/evidence/manifest.json`。
- 验证矩阵：`validation/summary.csv`；参数扫描：`exact_topk/`、`ivf_nprobe/`、`ivf_batch/`。
- 数据描述：`data/metadata.json`；建库信息：`index/build.performance.log`。
- 各次性能日志和 `cuda_*_sum.csv` 支持复算计时和 nsys 分析。
- 原始压缩包来源见仓库 `RESULTS_PROVENANCE.md`。文稿修订未新增 GPU 实验。

复现时应使用上述实验提交与已保存参数；当前文稿和排版文件可以独立更新。
完整构建和实验入口见 README，最小流程见报告附录 A。

### 随机种子记录

正文只说明固定种子与派生规则，具体取值在此保留：正式数据生成主种子为
`20260910`，正确性验证以 `20260825` 加上度量和精度偏移派生。
复现时以冻结配置与数据元信息为准；此次删去正文数字不改变任何实验输入。

### 图表重建

在任务根目录安装 `requirements.txt` 后，运行
`python3 scripts/redraw_report_figures.py`，从冻结的逐次 CSV 生成 8 图，
各保留 SVG、PDF 和 300 dpi PNG。输入 SHA-256、原始观测与汇总统计保存在
`report_assets/figure_data_manifest.json`。QPS 与逐次 P99 的误差条为运行间
样本标准差，不是置信区间；Exact 为 3 次、IVF 为 5 次。Recall 始终基于
同一组 20 条质量查询。新增内核时间图按各模式一次 nsys 记录中的评分与 CUB
排序累计时间归一化，未混入 CUDA API、内存传输或端到端时间。

## 数据来源与校验

生成器版本为 `clustered-synthetic-v2-chunked-fp32`，NumPy 为 1.26.4。
数据库和查询文件分别为 512,000,028 与 512,028 bytes。

| 文件 | SHA-256 |
|---|---|
| 数据库 | `f28d0b017c16de47dda3d1327695ba81bf861a240326b150304474520d299f10` |
| 查询 | `6c06c745295173a04a72f21221b3a18ecf0787d107b6df356ea5533fb1e0e27d` |

归档没有包含百万级数据库、查询及索引二进制。按保留元数据中的生成参数与种子
重新生成，不能将元数据存在描述为大文件已经归档。

## 训练营要求对照

| 内容 | 报告位置 | 主要实现或证据 |
|---|---|---|
| 精确检索与多 K | 第 3、6.2 节 | GPU 评分、CUB 排序，K=1 至 100 |
| 距离度量与输入格式 | 第 3、6.1 节 | L2/内积/余弦，FP32/FP16 输入 |
| 近似检索 | 第 3、6.3 节 | IVF-Flat，nlist=4096，nprobe 扫描 |
| 索引构建与持久化 | 第 4、6.1 节 | 聚类、CSR 倒排表、保存/加载 |
| 批量查询接口 | 第 4、6.4 节 | 分组参数扫描；当前内核逐查询同步执行 |
| CPU 参考与正确性 | 第 5、6.1 节 | 30 项验证、严格与容差排序判定 |
| 百万级规模 | 第 2、6 节 | 1M×128 数据、1000 条性能查询 |
| 质量和性能指标 | 第 5、6 节 | Recall、QPS、P50/P99、工作集与采样 CPU 比较 |
| Profiler 分析 | 第 7 节 | exact/IVF 的 nsys；NCU 因计数器权限不可用 |
| 优化方向与限制 | 第 8、9 节 | 主机候选处理、排序和串行调度；20 条质量查询限制 |

## 工具与文献核对记录

报告所引在线资料核对日期为 2026-09-19。文献用于设计背景和工具语义，
性能数字来自本项目实验。CUB 实测以 CUDA 12.8 下日志中的版本和内核符号为准。
NCU 返回 `ERR_NVGPUCTRPERM`，宿主机配置 `RmProfilingAdminOnly=1`；
这不影响已有 nsys 时间线，但不能据此补出硬件计数器结论。

### 扩充文献的来源与使用边界

本轮将正文参考文献由 4 项扩充为 12 项，并按首次引用顺序重新编号。
下列核对记录仅保留在仓库，不进入 PDF；未引用外部论文的性能数字作为本项目结果。

| 正文编号 | 文献与核对来源 | 核对要点与正文用途 |
|---|---|---|
| 2 | Douze 等，*The Faiss Library*：[作者预印本 v1](https://arxiv.org/abs/2401.08281v1)、[全文](https://arxiv.org/html/2401.08281v1) | 2024 年 v1；核对 9 位作者。用于检索设计空间、IVF 候选数近似及不均衡因素，不作为已运行的 Faiss 基准。 |
| 3 | Jégou、Douze、Schmid，*Product Quantization for Nearest Neighbor Search*：[Inria 论文清单](https://radar.inria.fr/report/2011/lear/bibliography.html)、[作者署名稿](https://www.researchgate.net/profile/Herve-Jegou/publication/47815472_Product_Quantization_for_Nearest_Neighbor_Search/links/00b4953c9a4b399203000000/Product-Quantization-for-Nearest-Neighbor-Search.pdf) | TPAMI 33(1), 117–128, 2011；DOI 10.1109/TPAMI.2010.57。DOI 中的 2010 不等于卷期年份；HAL 直取遇到机器人验证，未将失败访问记为全文核验。正文仅说明子空间量化与短码表示。 |
| 4 | Malkov、Yashunin，HNSW：[作者预印本 v4](https://arxiv.org/abs/1603.09320v4) | 明确引用 2018 年 v4，不混用首次预印本的 2016 年与后续期刊卷期。仅用于分层图导航背景，未实现或测试 HNSW。 |
| 5 | Aumüller、Bernhardsson、Faithfull，ANN-Benchmarks：[出版社页面](https://doi.org/10.1016/j.is.2019.02.006)、[作者预印本](https://arxiv.org/abs/1807.05614) | 期刊版本为 Information Systems 87, 101374, 2020；预印本为 2018 年。出版社检索记录核对卷、年、文章号；用于质量与效率联合比较，不声称使用其基准框架。 |
| 6 | Li 等，RadiK：[作者预印本与出版信息](https://arxiv.org/abs/2501.14336)、[全文](https://arxiv.org/html/2501.14336v1)、[作者实现](https://github.com/leefige/radik) | ICS 2024, 537–548；预印本上传于 2025 年。作者页面提供 DOI 10.1145/3650200.3656596。用于基数选择、阈值筛选与有序输出约束讨论，不声称已集成。 |
| 8 | Lloyd，*Least Squares Quantization in PCM*：[原始论文扫描件](https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf) | 原始页眉核对 IT-28(2), 1982；论文末尾延续至 137 页，故采用 129–137，而非部分二级目录中的 129–136。用于交替分配与中心更新的方法来源，不主张两轮迭代已收敛。 |
| 9 | NVIDIA：[CUDA C++ Best Practices Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/) | 使用与实测 Toolkit 对应的归档版。合并访存原则支持优化假设，不替代性能计数器测量。 |
| 11 | Amdahl，1967：[ACM 出版记录](https://doi.org/10.1145/1465482.1465560) | 核对作者 Gene M. Amdahl、会议、483–485 页及 DOI。用于其他阶段不变时的部分改进上界，不将内核占比与端到端占比混用。 |

原有 GPU 检索论文、CUB、Nsight Systems 与计数器权限说明分别移为编号 1、7、10、12。
新增设备工作集公式直接由 `src/search_cuda.cu` 的显式分配项推导，属于实现分析，
不是新增峰值显存测量。第 8.2 节的映射、设备端候选和选择算法对照均为待验证方案。

提交目录、分支、身份信息和文件清单以仓库 `SUBMISSION_GUIDE.md` 为准。
当前 PDF 按用户要求不署名，不展示提交号、修订日期或资料核对日期。
