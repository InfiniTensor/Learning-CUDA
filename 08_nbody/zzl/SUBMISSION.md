# 交付材料说明

## 项目内容

本项目用 CUDA 并行计算粒子间引力，模拟 N 体系统并输出可供 Python 读取的轨迹，
另有 CPU/OpenMP 对照实现。输入为粒子初态和积分参数，输出为二进制轨迹、性能日志、
最终状态及二维或三维动画。

## 材料清单

| 材料 | 路径 | 用途 |
|---|---|---|
| 使用说明 | `README.md` | 环境、构建、运行与验证命令 |
| 实验报告 | `REPORT.md` | 物理模型、实现方法、数值误差和性能结果 |
| 性能分析说明 | `docs/PROFILING.md` | Nsight Systems 采集条件与复现步骤 |
| 核心代码 | `src/`、`include/`、`CMakeLists.txt` | CPU/CUDA 模拟程序 |
| 数据与可视化 | `generate.py`、`trajectory.py`、`visualize.py` | 输入生成、轨迹读取和动画 |
| 测试 | `tests/` | 数值、文件格式、打包和性能分析回归 |
| 工具脚本 | `scripts/` | 基准测试、性能采集、报告生成和打包 |
| 示例 | `examples/`、`outputs/demos/` | 二体输入、示例动画及配套数据 |
| 实验记录 | `evidence/` | 验收结果、性能测量和工具日志 |
| Linux 程序 | `bin/nbody` | 交付包与仓库分支中的预编译程序 |
| 文件清单 | 交付包内 `MANIFEST.json` | 包内文件路径、大小和 SHA-256 |

交付包面向 WSL/Linux。目录 `bin/` 和包内 `MANIFEST.json` 由打包流程生成。
完整开发工作区保存各轮历史记录；正式交付包按上述清单选取材料。
CUDA Toolkit、Python 依赖和 Nsight Systems 按 README 与性能分析说明准备。

仓库分支包含上述代码、测试、工具脚本、文档、实验记录、示例动画与预编译程序；
`outputs/` 与 `delivery/` 中体积较大的运行产物由仓库的 `.gitignore` 排除，
随交付压缩包提供。包内 `MANIFEST.json` 记录该压缩包实际包含的路径、字节数和 SHA-256。

## 验证口径

数值验收覆盖二体与多体场景、CPU/CUDA 对照、积分收敛、尾块边界、轨迹格式及动画导出。
性能报告分别列出计算区间耗时、完整进程耗时和分析工具中的内核耗时。
三种计时覆盖范围不同，比较时使用同类指标。

包外同名 `.zip.sha256` 与 `.zip.verification.json` 用于核对压缩包版本和完整性。
运行测试记录与压缩包完整性记录分别保存，按各自记录的对象和验证范围解释结果。
