# GPU 向量检索项目提交核对表

本表只用于提交前核验。项目方法、结果和局限以 `FINAL_REPORT.md` 为准。

| 课程要求 | 实现与证据 | 状态 |
|---|---|---:|
| 二进制向量库、查询与参数输入 | `src/file_io.cpp`、`src/main.cpp`、`src/search_ivf.cpp` | 通过 |
| 有序 Top-K 输出 | Exact 与 IVF CLI 均输出 ID 和分数；相同分数按较小 ID 排序 | 通过 |
| CPU 正确性参考 | `src/cpu_search.cpp`、`tests/unit_tests.cpp` | 通过 |
| GPU 精确检索 | matrix 与 fused 两条路径，候选集合一致 | 通过 |
| 至少一种近似检索 | GPU IVF-Flat、索引保存与加载、recall@K 评测 | 通过 |
| K=1/10/50/100 | Exact 与 IVF 自动测试覆盖 | 通过 |
| 百万规模、128 维、1000 queries | Synthetic-1M 正式实验与原始 CSV | 通过 |
| 性能超过 CPU 同级实现 | Exact 相对单线程 FAISS CPU Flat 加速 73.5–87.3 倍 | 通过 |
| 查询性能与质量报告 | QPS、run/batch P50/P99、显存、recall 和误差 | 通过 |
| 平均距离/分数误差 | nprobe 扫描 CSV 与报告正确性章节均给出 rank 对齐 MAE | 通过 |
| FAISS 对照 | 同数据、K、batch、nlist、nprobe 和训练预算 | 通过 |
| 真实分布实验 | MNIST 60k 数据库、10k queries | 通过 |
| 自动测试与内存安全 | 新平台 CTest 18/18；Compute Sanitizer 0 errors | 通过 |
| 独立源码构建 | CUDA 12.8 全新目录从零构建，CTest 18/18，Exact/IVF 最小 GPU 查询通过 | 通过 |
| 可复现材料 | 固定随机种子、环境、命令、校验和、CSV 和图表 | 通过 |

提交目录应保留源码、头文件、测试、配置、脚本、报告、CSV 和 SVG；不提交
`build/`、原始数据、二进制索引、检索输出、profiler 二进制文件或远端凭据。
Agent 相关内容属于可选扩展，不纳入本次报告结论。

提交前仍需人工确认仓库路径满足课程规定的 `/<选题>/<学员 ID>/`。当前目录名
为 `PurplePulse`；该字符串仅表示学员 ID，在未获得目标 PR 路径前不做自动重命名。
