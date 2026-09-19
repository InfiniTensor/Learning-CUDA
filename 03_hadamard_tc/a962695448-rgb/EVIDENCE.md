# 验证证据与完整档案

本提交保留实现、必要测试、平台入口、简明报告及摩尔的核心复现材料。
完整开发分支和历史实验不随此次精简删除。

## 固定档案

- [完整归档提交 3676727](https://github.com/a962695448-rgb/Learning-CUDA/tree/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb)
- [历史 results 目录](https://github.com/a962695448-rgb/Learning-CUDA/tree/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results)

该 results 目录的 2,586 个历史文件通过固定提交引用。日志、失败记录、数据和
实验脚本仍可在个人仓库查询；本 PR 的功能和测试不要求下载全部历史归档。
复现某次历史实验时，应使用该实验清单指定的源码、环境和输入；历史说明中的
results 路径属于完整档案检出。新运行可以在当前目录创建自己的结果目录。

## 重点审查入口

| 内容 | 入口与范围 |
|---|---|
| NVIDIA 最新生产验证 | [A100/4090 调用方式与自动规则](reports/a100-calling-validation.md)，原始数据链接固定到对应档案 |
| 独立参考与线程配置 | [A100 跨卡报告](reports/a100-validation.md)，包含固定 Dao 对照与全部负例 |
| 小维度行打包 | [A800/4090 报告](reports/packed-rows-validation.md)，按设备和 Graph/普通调用分别说明 |
| N256 显式融合布局 | [4090](reports/fused-layout-validation.md)、[A100](reports/fused-layout-a100-validation.md) |
| 五种国产平台 | 各 platforms 子目录 README，分别列出 SDK、设备、正确性和本机基线 |
| 摩尔原始计时与再生成 | [公开复现材料](platforms/moore/repro/README.zh-CN.md)，含 6,750 条计时、冻结对照和夹具生成器 |
| 代码来源与本轮检查 | [SUBMISSION_MANIFEST.json](SUBMISSION_MANIFEST.json) |

## 首轮精简提交的验证边界

初次精简阶段没有改变 CUDA/MUSA 等计算源码。保留代码的 Git blob SHA 与固定档案一致；
当时的工作是精简提交、整理文档链接和执行 CPU 检查。此前 GPU 结果仍归属于
其清单明确记录的实测版本，不将本轮整理写成新的 GPU 实验。

本地 CPU reference 检查、摩尔离线复算及数据完整性检查通过。原始 GPU 结果中
测试范围、跳过项、慢例及计时方式均保留，跨设备及不同计时口径不直接相除。

## 2026-09-17 独立量化入口

新增显式 packed 量化入口的完整范围见 [报告](reports/quantize-packed-20260917.md)。
[固定档案](https://github.com/a962695448-rgb/Learning-CUDA/tree/29bb34a0b2817e282a9554bd92c3911a3c010762/03_hadamard_tc/a962695448-rgb/results/quantize-packed-20260917) 包含最终双重性能门槛、正确性检查、源码和复现准备工具，也保存新增参数引入主机开销的初版实验。当前源文件与新测试哈希已在 SUBMISSION_MANIFEST.json 更新；未将先前的跨卡或国产平台记录当成本轮新入口的实测证据。

## 2026-09-17 单元素穷举与未采用候选

[报告](reports/singleton-quantization-20260917.md) 记录当前生产内核的完整单元素位型检查，以及未达收益门槛的符号简化候选。生产计算源码未变，新工具的指纹已登记；失败候选的源码和计时数据独立保留。

[本轮固定档案](https://github.com/a962695448-rgb/Learning-CUDA/tree/5e80eadd156c82a299e7cc7991bf46d1904243e6/03_hadamard_tc/a962695448-rgb/results/singleton-quantization-20260917) 保存完整原始结果与可验证的准备工具。

## 2026-09-17 成对量化

[最终报告](reports/paired-quantization-20260917.md) 记录 RTX4090 的分量编码、偏移、旧接口、长采样和 A/A 校准。新增 kernel 与测试的哈希已登记；三版源码及失败协议分别归档，不借用别的型号或历史源码作为本轮验证。

[本轮固定验证档案](https://github.com/a962695448-rgb/Learning-CUDA/tree/9b3bb79f3c67b6c5e284be7ded9559148b5ae448/03_hadamard_tc/a962695448-rgb/results/paired-quantization-20260917) 保留完整的所有阶段及原始判定。

## 2026-09-17 打包变换验证与成对候选

[本轮报告](reports/paired-hadamard-20260917.md) 列出基线验证与两个 REJECT 候选。计算/构建源码保持 b49fff4，新增验证器按实机通过的内容登记指纹；完整两版数据和源码通过报告中的固定归档链接复现。

## 2026-09-17 普通调用交叉诊断

[诊断报告](reports/host-attribution-20260917.md) 区分独立中位数与配对统计，并给出同源模块、缓冲池和测量顺序对照。固定证据通过报告链接访问；本次不新增生产计算变化或采纳结论。

## 2026-09-17 整合入口完整验证

[本轮报告](reports/integrated-pairs-20260917.md) 保存CLI六配置自测、接口回归和588设备/588主机对照。唯一未过判据为绝对耗时离散度；相对性能达标不自动覆盖该失败。原始数据可通过无损恢复工具逐文件校验。

## 最后一轮：对齐读写候选已采用

[验收报告](reports/wide-pairs-io-20260918.md)；[固定原始数据与预登记源码](https://github.com/a962695448-rgb/Learning-CUDA/tree/89dbadfa47197c9ada14007c7cc18039f32c1302/03_hadamard_tc/a962695448-rgb/results/wide-pairs-io-20260918)。三轮ACCEPT，254冻结输入一致，1050设备/2064主机/24目标组/18同源校准全部通过。早前负结果仍保留原判定。
