# 报告证据与提交对照

本文件保存从 PDF 中移出的版本、证据路径和提交管理信息，不属于报告正文。
报告源文件为 `../REPORT.md`；构建记录位于仓库 `output/pdf/report_build_manifest.json`。

## 实验来源

- 正式 GPU 实验提交：`6a6b582d2c491fe110bfc27972353c6e8bcd81cb`，运行时工作树 clean。
- 本地完整归档：`output_task/task-05-derivatives-pricing/task5_report_6a6b582/`（相对仓库根目录）。
- 随报告提供的轻量证据：`../report_assets/evidence/`。
- 证据清单及 SHA-256：`../report_assets/evidence/manifest.json`。
- 正式验证：`validation/summary.csv`；收敛与性能数据分别保存在
  `european_mc/`、`asian_mc/`、`barrier_mc/`、`european_qmc/` 和 `asian_performance/`。
- 原始压缩包来源见仓库 `RESULTS_PROVENANCE.md`。文稿修订未新增 GPU 实验。

复现时应使用上述实验提交与已保存参数；当前文稿和排版文件可以独立更新。
完整构建和实验入口见 README，最小流程见报告附录 A。

### 随机种子与多卡验证范围

正文保留种子派生与独立参考的设计，不列出具体种子数字。正式实验的主种子
与验证种子为 `20260824`；亚式与障碍 CPU 参考种子为 `4738792141952715314`。
逐次派生结果以冻结 CSV/JSON 和原实验配置为准，不因文稿调整而更改。

全部正式实验数据来自单张 RTX 5090。第 4.3 节仅保留未经多卡运行验证的
扩展设想；现有接口不等于多卡实测证据，合并估计、分区独立性和逐卡 LSM
策略均待验证，没有多卡加速比或扩展性结论。

### 图表重建

在任务根目录安装 `requirements.txt` 后，运行
`python3 scripts/redraw_report_figures.py`。脚本逐组复算 10 次运行的均值，
并与冻结的 `aggregate.csv` 核对；不重新执行定价实验。每图导出 SVG、PDF 与
300 dpi PNG，输入 SHA-256 和逐次统计保存在 `report_assets/figure_statistics.json`。
标准误差图的带状区域是运行间样本标准差；绝对差图为第 25–75 百分位区间，
均不表示均值置信区间或参考价格的不确定性。RQMC 图中的普通 MC 来自对应
欧式正式实验，保留各自真实样本数，不对齐或补造测点。

## 训练营要求对照

| 内容 | 报告位置 | 主要实现或证据 |
|---|---|---|
| 输入参数解析 | 第 2 节 | `src/config.cpp` |
| 欧式与路径依赖产品 | 第 3、6 节 | 欧式、亚式、障碍正式扫描 |
| GPU 并行粒度与内存组织 | 第 4 节 | `src/pricing_cuda.cu`，256 线程与两阶段归约 |
| 误差及收敛实验 | 第 5、6 节 | 多规模、10 次重复与 8 幅曲线 |
| 解析及 CPU 参考 | 第 6.1 节 | BS/CRR、32 项检查，其中 16 项有独立参考 |
| 价格及性能输出 | 第 7 节 | `src/main.cpp`、7 次性能测量 |
| 方差缩减与 RQMC | 第 6.2、6.3 节 | 反向变量、控制变量与随机化 Halton |
| Greeks、美式及复杂模型扩展 | 第 3、6.1、9 节 | 后端一致性与独立参考的覆盖范围分别说明 |
| Profiler 分析 | 第 7.2 节 | 有效 nsys；NCU 因计数器权限不可用 |
| 实测平台 | 第 2、9 节 | NVIDIA RTX 5090 单卡，未提供国产平台或多卡性能结果 |

## 环境排错记录

初次配置曾因 `nvcc` 不在 PATH 而使用 CPU 替代后端，后续显式指定
`/usr/local/cuda-12.8/bin/nvcc`，并核对编译命令中的 `pricing_cuda.cu` 与
运行结果的 CUDA 后端标签。NCU 返回 `ERR_NVGPUCTRPERM`，宿主机配置
`RmProfilingAdminOnly=1`；保留原始日志，不把日志文件存在视为成功采集。

提交目录、分支、身份信息和文件清单以仓库 `SUBMISSION_GUIDE.md` 为准。
当前 PDF 按用户要求不署名，不展示提交号、修订日期或资料核对日期。

## 正文新增引用与支撑位置

下列编号与 `REPORT.md` 参考文献一致；书目信息按出版社、作者页面或作者所在
机构记录核对。引用用于说明数值方法及其适用条件，不构成新增实验结果。

| 编号 | 文献及已核对书目信息 | 正文支撑位置 | 原始来源 |
|---|---|---|---|
| 7 | Salmon、Moraes、Dror、Shaw，SC '11，Article 16，1–12；DOI `10.1145/2063384.2063405` | 第 3.3 节：Philox 的计数器型设计、应用索引与并行随机流 | [作者网页](https://random123.com/)；[作者提供的全文](https://random123.com/papers/random123sc11.pdf)，第 1 节明确讨论以应用变量而非机器参数组织随机流 |
| 8 | Halton，1960，Numerische Mathematik 2，84–90；DOI `10.1007/BF01386213` | 第 3.3 节：Halton 基数反演点集的方法来源 | [Springer 原文条目](https://link.springer.com/article/10.1007/BF01386213)，核对作者、题名、出版年、卷、页码与 DOI |
| 9 | Cranley、Patterson，1976，SIAM Journal on Numerical Analysis 13(6)，904–914；DOI `10.1137/0713071` | 第 3.3 节：随机平移思路及按随机化重复估计积分误差 | [SIAM 出版社条目](https://epubs.siam.org/doi/10.1137/0713071)，摘要说明随机化积分规则与误差区间；采用纸本年 1976，而非网页上线年 2006 |
| 10 | Broadie、Glasserman，1996，Management Science 42(2)，269–285；DOI `10.1287/mnsc.42.2.269` | 第 3.4.3 节：路径导数、似然比与重复模拟的区别 | [INFORMS 出版社条目](https://pubsonline.informs.org/doi/10.1287/mnsc.42.2.269)；[作者提供的全文](https://www.columbia.edu/~mnb2/broadie/Assets/bg_ms_1996.pdf) |
| 11 | Lord、Koekkoek、van Dijk，2010，Quantitative Finance 10(2)，177–194；DOI `10.1080/14697680802392496` | 第 3.4.1、9.1 节：负方差处理影响偏差；本文投影方案与完全截断不同 | [作者机构正式发表记录](https://repub.eur.nl/pub/18571)；[Tinbergen 作者稿](https://papers.tinbergen.nl/06046.pdf)，第 4 节及表 1/2 对比修正规则，完全截断保留可能为负的辅助状态 |
| 12 | Boyle、Broadie、Glasserman，1997，Journal of Economic Dynamics and Control 21(8–9)，1267–1321；DOI `10.1016/S0165-1889(97)00028-6` | 第 1、3.2、3.4.3、9.1 节：效率研究定位、反向变量条件、CRN 有限差分及同样本控制系数估计 | [Elsevier 出版社条目](https://www.sciencedirect.com/science/article/pii/S0165188997000286)；[论文全文](https://www.cs.hunter.cuny.edu/~felisav/StochasticProcesses/Course_Material_files/BBG_jedc96.pdf)，第 2 节讨论方差缩减，第 4.1 节讨论共同随机数差分 |

文献与实现的区别已写入正文：cuRAND 执行实际 Philox 采样，未另行实现
Random123 库；随机平移施加于 Halton 点集，未宣称实现原始 Korobov 格点规则或
数字扰乱；路径导数与似然比仅作为后续比较对象；Heston 保留本文投影递推，
未将完全截断方案的收敛或性能结论移用于当前实现。
