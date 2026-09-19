# User Requirements

- 任务五所有公开功能都要在 CPU 与 GPU 得到准确结果，用于最终对比。
- 本地仅运行 CPU；GPU 构建、测试、profile 和性能实验由用户通过 SSH 在服务器完成。
- 关键函数使用中文注释，解释金融假设、随机数、数值方法和并行归约。
- 报告只采用可复现的原始输出，不把 smoke test 或未运行的 GPU 路径写成最终结果。
- 正式实验使用 clean commit 的 `report` preset；`quick` preset 只检查流水线。
- 数值 reference 与 CPU 单线程性能 baseline 分开记录，禁止混用两者的路径数和计时。

### Document Preferences

- 语言：中文正文 + English 标题。
- 收敛、方差缩减、CPU/GPU gate 和性能结果分表呈现。
- 保留 seed、环境、命令和逐案例原始日志。
