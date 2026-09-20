# 题目 2 提交说明

训练营 ID：**曹泽阳**；GitHub ID / 目录名：`zmuxuny`。

| 提交项 | 本题内容 |
|---|---|
| fork | [zmuxuny/Learning-CUDA](https://github.com/zmuxuny/Learning-CUDA) |
| 上游目标 | `InfiniTensor/Learning-CUDA:2026-summer-project` |
| PR 分支 | `projects/quant-dequant` |
| 目录 | `02_quant_dequant/zmuxuny/`，可独立构建，不依赖另一题 PR |
| PR | [#80](https://github.com/InfiniTensor/Learning-CUDA/pull/80) |
| 标题 | `【训练营】【题目2】MXFP8/NVFP4 软件量化与反量化（多平台）` |
| 总结报告 | [REPORT.md](REPORT.md) |
| 使用与测试 | [README.md](README.md)；`tests/` |

## 要求与材料对应

依据《2026夏季训练营 CUDA 方向项目》开头“项目代码要求”及题目 2“需提交内容”：

| 要求 | 对应内容 |
|---|---|
| FP32/FP16 输入与 FP16/BF16/FP32 输出 | README 配置表、文件协议、独立重载示例 |
| MXFP8/NVFP4、真实位宽打包、缩放与舍入 | REPORT 数值定义；include/；tests/validate.py |
| 分布误差、压缩率、kernel 时间、有效带宽 | REPORT 误差与性能表；results/ 各平台原始 JSON |
| NVIDIA 必需、国产适配加分 | REPORT 平台表及对应平台实验报告 |
| ncu 和/或 nsys 使用与分析加分 | PROFILING.md 与原始日志；Sanitizer 未列为必需项 |

开头通用说明的“无测试代码”与题目专门条款的“包含测试”存在文字冲突；本提交按题目专门条款保留 `tests/`，正式程序位于 `src/`、`include/`、`ascend/`。题目没有强制每题单独 PR，本提交分开以便独立评审。

`PR_BODY.md` 为完整 PR 正文；`REPORT.md` 为总结报告，各 `REPORT_<平台>.md` 给出最终实现与实测结果。
