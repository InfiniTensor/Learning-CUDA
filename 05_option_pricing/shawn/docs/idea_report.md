# Idea Report

## Part 1 — Motivation

任务五以 Monte Carlo 衍生品定价为核心，目标是在同一工程中建立可解释 CPU reference、CUDA 并行实现、统计误差与性能证据链。

### Research Questions

- RQ1：各产品/模型在 CPU 与 CUDA 上能否在联合统计容差内一致？
- RQ2：方差缩减与 QMC 如何影响误差、标准误和吞吐？
- RQ3：GPU kernel、归约与多 GPU 分区分别带来多少性能收益？

### Key Works

| 方法 | 工程用途 |
|---|---|
| Black–Scholes | European 解析价格与 Greeks reference |
| Heston | 随机波动率路径模型 |
| Longstaff–Schwartz | American 提前行权回归 |
| Randomized QMC | Halton shift replication 与标准误估计 |

## Part 2 — Method

CPU/CUDA 共享产品、模型、方差缩减、随机数与有限差分口径；价格门禁使用联合标准误，Greeks 使用显式绝对/相对容差。American LSM 的小型线性方程由 host 求解，其余路径与回归矩由 CUDA 处理。

## Part 3 — Experiment Design

主实验包含 GBM/Heston/Local Vol、European/Asian/Barrier/American、pseudo/Halton、Greeks 和可选多 GPU。每个报告结论保存配置、seed、环境、原始结果、comparison CSV 和 profiler 文件。

American LSM 的有限差分 Gamma 对路径噪声和提前行权边界更敏感，正式伪随机 CPU/GPU gate 因此使用 200,000 条路径和 1% spot bump；其他模型仍使用快速 smoke 配置。Halton 案例用于验证共享低差异序列下的 CPU/GPU 实现一致性，但只有在同时对照 CRR 树或其他独立 reference 后，才作为 American Gamma 的绝对精度证据。

正式 accuracy matrix 将“后端一致性”和“独立 reference 一致性”同时记录：Black–Scholes European 的价格与 Greeks 对照解析解，Black–Scholes American 的价格与有限差分 Greeks 对照 2,000 步 CRR 树。Heston 与 Local Vol 暂无独立闭式 reference，矩阵只对它们执行 CPU/GPU gate，并在报告中明确这一边界。American Halton 使用独立于通用 smoke 的更高路径数配置，避免把低样本 QMC 的二阶差分噪声误当作正确性结论。

## References

- Black and Scholes (1973), *The Pricing of Options and Corporate Liabilities*.
- Heston (1993), *A Closed-Form Solution for Options with Stochastic Volatility*.
- Longstaff and Schwartz (2001), *Valuing American Options by Simulation*.
