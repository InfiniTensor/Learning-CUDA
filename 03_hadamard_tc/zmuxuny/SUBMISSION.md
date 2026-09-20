# 题目 3 提交说明

训练营 ID：**曹泽阳**；GitHub ID / 目录名：`zmuxuny`。

| 提交项 | 本题内容 |
|---|---|
| fork | [zmuxuny/Learning-CUDA](https://github.com/zmuxuny/Learning-CUDA) |
| 上游目标 | `InfiniTensor/Learning-CUDA:2026-summer-project` |
| PR 分支 | `projects/hadamard` |
| 目录 | `03_hadamard_tc/zmuxuny/`，可独立构建，不依赖另一题 PR |
| PR | [#81](https://github.com/InfiniTensor/Learning-CUDA/pull/81) |
| 标题 | `【训练营】【题目3】Hadamard 矩阵加速与量化融合（多平台）` |
| 总结报告 | [REPORT.md](REPORT.md) |
| 使用与测试 | [README.md](README.md)；`tests/` |

## 要求与材料对应

依据《2026夏季训练营 CUDA 方向项目》开头“项目代码要求”及题目 3“需提交内容”：

| 要求 | 对应内容 |
|---|---|
| FP16/BF16、多种头维度、误差阈值 | README 输入约定；REPORT 正确性；tests/validate.py |
| 融合与非融合一致 | 各路径 packed/scales/global scale 逐字节检查 |
| 变换性能与矩阵加速对照 | REPORT 同版本蝶形/矩阵、融合/非融合性能表 |
| NVIDIA 必需、国产适配加分 | REPORT 平台表及对应平台实验报告 |
| ncu 和/或 nsys 使用与分析加分 | PROFILING.md 与原始日志；Sanitizer 未列为必需项 |

开头通用说明的“无测试代码”与题目专门条款的“包含测试”存在文字冲突；本提交按题目专门条款保留 `tests/`，正式程序位于 `src/`、`include/`、`ascend/`。题目没有强制每题单独 PR，本提交分开以便独立评审。

`PR_BODY.md` 为完整 PR 正文；`REPORT.md` 为总结报告，各 `REPORT_<平台>.md` 给出最终实现与实测结果。
