# 提交整理与补充建议

## 已完成的结构调整

以参考作业的五个主要目录 `src/`、`include/`、`scripts/`、`assets/`、`test_images/` 为格式，重构本人的 NLM 提交。仓库外层的季度说明、选题目录和 LICENSE 保持原有内容。

| 原位置/内容 | 新位置/处理 |
|---|---|
| 多层 `include/{core,nlm,pal,kernels,tester}` | 扁平 `include/` |
| 四平台 `kernels/` 与共享 `.inl` | `src/kernels.cu/.maca/.mu`；CUDA 和 CoreX 共用 `.cu` |
| `tester/` | `src/benchmark.cpp`、`metrics.cpp`、`validate.cpp`、`self_check.cpp` |
| `third_party/stb/` | `include/`，保留头文件内完整许可证 |
| `params/default.txt` | 根目录 `params.txt` |
| 多份相同默认参数 | 合并到 `params.txt`；其余位于 `scripts/params/` |
| `docs/` 多篇重叠说明 | `README.md` 与 `lab_report.md/.pdf` |
| `docs/assets/` | `assets/` |
| `data/clean/`、`data/noisy/` | `test_images/`，新增来源说明 |
| 四份平台复现脚本 | 统一 `scripts/reproduce.sh` |
| 个人路径 WSL 脚本、过程记录、重复输出图片 | 移出提交；保存在仓库外完整备份中 |
| `experiments/results/` | 精选 CSV、环境、验证与 trace 位于 `assets/results/` |
| 所有运行输出 | `output/`，由 `.gitignore` 排除 |

`make run` 现执行降噪；`make test` 保留自检与 GPU 验证；默认 `make` 只构建。编译对象按平台隔离，头文件及编译选项变动会触发重编。脚本使用 LF 换行。

## 建议补充的文件（按优先级）

| 优先级 | 建议文件 | 应提供的实质内容 | 当前状态 |
|---|---|---|---|
| P0 | `assets/review_response.md` | PR 链接、逐条评语、对应修改和验证证据 | 原 PR #68 已关闭且未合并；尚未取得具体评语 |
| P0 | `assets/algorithm_contract.md` | 题面 dist 定义、2*sigma² 与 2*sigma²*N_patch 的关系、h 标定、RGB/边界处理、教师认可口径或严格题面验证结果 | 已在报告揭示差异，尚需解决 |
| P0 | `assets/results/<device>/environment.txt` 与新 CSV | 最终 commit、工作树状态、编译选项；NVIDIA 1080p/4K mean/min/stddev 与同条件 CPU 重复数据 | 旧 NVIDIA 主要为 legacy 口径 |
| P1 | `test_images/dataset_manifest.csv` | 自然图来源、授权、分辨率、通道、SHA256、clean/noisy 对应、噪声生成参数 | 当前只有合成示例说明 |
| P1 | 独立正确性样例与验证结果 | 小尺寸手算例、边界/纹理图、标准差受控的噪声；与题意独立对照，避免只证明 CPU/GPU 相互一致 | 现有 12 项自检与 CPU/GPU 验证保留 |
| P1 | `assets/nsight/` | 原生 nsys 时间线和 ncu 的吞吐、占用率、寄存器/内存指标截图，附命令和环境 | 有 nsys 文本与图表，缺 ncu 计数器 |
| P2 | `assets/demo.mp4` | 参数选择、输入/输出、可重复的运行演示 | 尚未制作，不影响代码构建 |

不要创建空白占位文件来代替证据。自然图的使用授权和结果需真实提供；新的时间线截图应来自本人实现，不能使用参考作业截图。

## 优先解决的内容问题

1. 当前噪声生成器使用均匀噪声幅度，题面 sigma 是标准差估计。报告已说明区别；若改为高斯噪声或标准差标定，应更新文件名/元数据并重跑质量实验，不能直接沿用旧指标。
2. 当前公式减项按 patch 大小和通道数缩放。CPU 与 GPU 一致不能单独证明满足题意。先确定算法口径，再进行正式重跑。
3. `e2e_ms` 不包含文件 I/O 和函数退出时的显存释放，吞吐量按 kernel 计算。不要再将其称为完整应用耗时或直接声称达到完整 4K 交互目标。
4. 旧 NVIDIA 的 kernel min 与其他平台的 kernel mean 不可混排；缺少采样值时不补造标准差。
5. 主核耗时占比高只说明优化对象，不能据此宣称 compute-bound；ncu 数据仍需补采。

## 本次验证记录

2026-09-16，在 WSL Ubuntu、RTX 3060 Laptop、CUDA 12.9、sm_86 上实际执行：

| 检查 | 结果 |
|---|---|
| 全部迁移源码干净构建 | PASS |
| Host 源码 C++11 语法兼容检查 | PASS，`g++ -std=c++11 -Iinclude -fsyntax-only src/*.cpp` |
| `make test`：12 项 Host 自检 + 256×256 RGB 三版本对 CPU | PASS，V0/V1/V2 MAE=0、PSNR=inf |
| `make run` 默认图像处理 | PASS，写入 output/images/ 与 output/logs/ |
| 相同配置再次构建 | PASS，kernel 对象时间戳不变，无重复编译 |
| NVIDIA / Iluvatar / MetaX / Moore 构建命令检查 | PASS；后三平台仅检查构建依赖和命令生成 |
| 所有保留 Bash 脚本语法检查 | PASS |
| 64×48 全流程复现：构建、验证、24 组合、CPU、质量、证据哈希 | PASS；本机 profiler 缺失，明确标记 SKIPPED |
| 1080p clean/noisy 图重新生成 | 与保留样例逐字节一致 |
| 1080p V2/base 输出 | 与保留的历史 MetaX 输出逐字节一致 |
| 31×23 灰度图（非整块边界）三版本对 CPU | PASS，MAE=0、PSNR=inf |

本地日志保留在被忽略的 `output/refactor_validation/`，可按需审阅；小图完整复现仅验证脚本链路，不替代 1080p/4K 正式性能验收。国产平台需在对应设备上重跑，历史日志不替代本次重构验证。

## 提交前命令

```bash
make build PLATFORM=nvidia ARCH=sm_89
make test PLATFORM=nvidia ARCH=sm_89
PLATFORM=nvidia ARCH=sm_89 bash scripts/reproduce.sh
python3 scripts/export_report.py
git diff --check
git status --short
```

确认 PR 目标分支为 `2026-summer-project`，变更范围仅包含 `07_nlm_denoise/blackbook537/`。

原提交为 [PR #68](https://github.com/InfiniTensor/Learning-CUDA/pull/68)，目前已关闭且未合并。此次更新沿用 `blackbook537/Learning-CUDA-2026` 的 `submit/nlm-denoise-blackbook537` 分支。重新提交时，应说明本次结构整理与验证范围，保留上述尚需补充的内容，不将历史跨平台数据写成本次重新实测。
