# 核心目录迁移验证

日期：2026-09-10。此次变更只整理正式代码归属、构建/脚本路径并抽出已有编码辅助，不是性能优化实验。

## 变更边界

- 正式 pipeline、输出模块、CPU 桥接、Python 工具和测试迁入 Core。
- 新增统一 CMake 入口；旧 CUDA CMake 通过转发文件仍可构建原目标。
- 新 GPU 编译单元不再包含旧 CUDA main.cu，改用 codec.cuh 的相同编解码规则。
- CPU reference 源码和冻结数据保留原位置；配置原路径保留，新实验使用 Core/configs。
- benchmark 环境哈希现在记录 Core 源码/配置/构建定义和两个外部 CPU main.cpp；旧实验文件不重写。

## 已执行检查

| 检查 | 结果 |
|---|---|
| Release、默认 sm_75 配置/构建 Core 两个目标 | 通过，CUDA 12.0、GCC 13.3 |
| Core CTest | 7/7 通过：工具 1 项，两格式各文件测试、算法回归、冻结哈希 |
| 原 CUDAMXFP8 全部目标重新构建及 CTest | 29/29 通过 |
| 原 CUDANVFP4 全部目标重新构建及 CTest | 29/29 通过 |
| 冻结输入/golden SHA256 | 两格式各 15 项全部匹配 |
| 迁移前旧二进制与新 Core 二进制对照 | 24 组全部匹配，见下文 |
| 根目录旧 Python run 命令 | 两种格式通过，转发后使用 Core 构建产物 |
| benchmark/profile 脚本迁移测试 | 模拟外部工具，路径、报告生成、哈希及缺失 GPU 数据的失败分支通过 |
| 两种新二进制 --benchmark 1025 1 | 实际 GPU 冒烟运行通过，输出量化/三类反量化/host_api 指标；不作为正式性能样本 |
| git diff --check | 通过 |

数值迁移对照在专用临时目录生成 33 元素 FP16 outlier 输入，组合为 2 格式 × 2 scale 模式 × 2 舍入 × 3 输出类型，共 24 组。分别运行迁移前备份的 pipeline 二进制和 Core 新二进制，检查：

- .lpq 文件逐字节相同，包括 header、packed data、scales。
- 反量化输出文件逐字节相同，包括 FP32、FP16、BF16。
- MAE、MSE、最大绝对误差、两类压缩率和 CPU 对照字段相同。
- 时间/带宽字段不要求相同，也没有据此计算加速比。

迁移前源码和二进制临时备份位于 `/tmp/core-migration.VxrHzm`，仅供本次核对，不作为长期版本管理。长期复现应保存完整源码或人工审查后提交 Git；本次未提交或推送。

## 未执行事项

此次未重新完成正式 20 次性能采样、实际 nsys 采集或 Compute Sanitizer 检查；工具脚本模拟测试不能代替这些实际 GPU 检查。旧结果也不自动视为迁移后结果。命令见 README 第 9 节。

旧 CUDA 可分离编译目标链接时出现系统静态库 incompatible 跳过警告，构建和全部测试成功；Core 新目标构建成功。此处保留事实，不把警告解释为算子数值错误。
