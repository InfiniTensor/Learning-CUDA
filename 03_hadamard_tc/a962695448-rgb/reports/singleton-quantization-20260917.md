# 单元素 INT4 穷举与性能实验（2026-09-17）

## 保留结果

生产计算内核保持基线 `f9da76087f6cbf6aad52e9effa6cf3ac3aabccf0` 的字节版本。本轮提交 `scripts/verify_singleton_quantization.py` 穷举检查和可复现实验记录，未纳入收益不足的计算特化。

FP16 和 BF16 各有 65,536 种 16 位存储编码；本轮覆盖共 **131,072 种编码**，其中 **128,768 个有限值**（FP16 63,488、BF16 65,280）与独立 NumPy INT4 参考逐位一致。范围仅为 N=1 的独立量化，不是任意长度向量的穷举。

二维 `(65536,1)` 以及四维尾行 `(1,1,65539,1)`、128/256 线程、两个 dtype 组合共 8 组。每组检查 allocating/out 一致、packed 字节与 float32 scale、输入不变、输出哨兵、存储地址、版本计数及 N=1 的高半字节为零。

同一运行分别检查原生产版本与候选。非有限编码只与**同一旧 packed 路径**对照以观察回退一致性，不扩展 API 约定的有限值输入域；其他布局对非有限编码可能有不同结果。独立脚本未传入旧扩展时，会在报告中把这项旧版本比较标为 NOT_CHECKED，不能冒充独立验证。

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=1 python scripts/verify_singleton_quantization.py \
  --build-directory /tmp/hadamard-singleton-check \
  --report /tmp/hadamard-singleton-check.json
```

## 未采用的候选

候选对 packed 的 N=1、Transform=false 分支使用符号直接确定有限非零输入的 ±7 量化码，零得到 0；scale 仍沿用原公式，非有限输入回退到原浮点运算。该候选通过完整穷举，以及原 700 组 packed 检查、16 组流/Graph、372 正例/63 负例/4 项 mutation 检查。

性能以**当前已经优化的 packed 实现**为基线，使用与正式测量分开的正确性阶段。RTX4090D、CUDA 12.8.61、Torch 2.6.0a0+ecf3bae40a.nv25.01、NumPy 1.26.4：

- 目标为 65,536 / 262,147 / 1,048,579 行，N=1，FP16/BF16，128/256 线程。
- 对照为小行数、其他 packed 小维度、原量化、Hadamard 及融合入口。
- 三轮共 **132 条设备、132 条普通调用**成对记录；每条保留七个配对样本。设备为 64 节点 Graph × 10 次重放的 CUDA event 时间，主机调用另外同步计时。
- 12 个 dtype/线程/轮次分组的目标几何平均只有 **1.0266–1.0437×**；单配置目标范围 **0.9987–1.0820×**。普通 allocating/out 目标整体几何平均约 **1.0151 / 1.0144×**。
- 每场景“不退化超过 5%”通过，最大设备/主机退化分别为 **0.466% / 2.667%**；但全部目标分组都未达到预先设定的 **1.10×** 门槛，因此判为 **REJECT**，保持现有生产内核。

此处没有将 2.7%–4.4% 的局部设备收益写成全面加速，也没有在看到结果后降低门槛。[候选源码、原始数据、原生产版本穷举结果和复现工具](https://github.com/a962695448-rgb/Learning-CUDA/tree/5e80eadd156c82a299e7cc7991bf46d1904243e6/03_hadamard_tc/a962695448-rgb/results/singleton-quantization-20260917) 已固定归档；本轮新增验证工具不依赖采用该候选。
