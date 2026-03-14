## 简介

最新研究表明，在量化前对激活进行随机旋转（如 Hadamard 变换）可以有效抑制异常值，提升低比特量化精度——这是 QuaRot、SpinQuant、FlashAttention-3 等前沿工作的核心思想。然而，Hadamard 变换本身引入了额外计算开销，可能抵消量化带来的收益。

公司要求你实现一个利用 Tensor Core 加速的 Hadamard 变换核，将旋转开销降至最低，使得 FP8/INT4 量化在保持精度的同时真正加速。

## 任务内容
实现一个基于 Tensor Core 的快速 Hadamard 变换核，支持多种输入尺寸，并与量化算子融合。

## 📬 有疑问?

更多详细信息和要求请参考本季度项目文档。

可以在项目群里直接询问导师和助教！

Good luck and happy coding! 🚀
