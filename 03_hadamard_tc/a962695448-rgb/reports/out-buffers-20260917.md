# 可复用输出接口与普通 Python 调用性能

新增 `hadamard_out`、`hadamard_int4_out`、`quantize_int4_out`，用于形状稳定、能够复用输出的前向调用。接口写入调用者的缓冲区并返回 `None`；原分配式接口和默认参数保持不变。

## 用法

```python
y = torch.empty_like(x)
packed = torch.empty((*x.shape[:-1], (x.shape[-1] + 1) // 2), device=x.device, dtype=torch.uint8)
scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)

op.hadamard_out(x, y)
op.hadamard_int4_out(x, packed, scales)
op.quantize_int4_out(y, packed, scales)
```

输出必须具有精确形状、dtype、相同 CUDA 设备和连续布局，不会自动 resize。输入与各输出的活动字节区间必须互不重叠，同一分配中的不重叠视图允许使用。再次调用会覆盖旧输出；需要保留旧结果时先复制。跨流使用、复用和释放缓冲区时由调用者管理同步与生命周期。

输入和输出均遵守前向约束，不能 requires_grad。写入维护 PyTorch 版本计数，能够识别其他计算为反向传播保存的数据已经被修改；inference tensor 须在 inference_mode 内更新。CUDA Graph 重放遵循 PyTorch 的捕获规则，主机端版本更新发生于调用/捕获阶段。

## 验证

| 检查 | 结果 |
|---|---|
| 正常调用 | 372 组配置，每组两份输入，与独立 NumPy 参考逐位一致 |
| 拒绝行为 | 63 项异常在写入和版本变化前拒绝，含 DLPack/部分/输出间重叠 |
| 存储与修改记录 | 输出地址和哨兵保持；保存给反向的数据能检测修改；inference_mode 通过 |
| 输出分配 | 预热后重复 50 次调用，无额外 PyTorch 输出张量显存分配 |
| 执行上下文 | 24 组流/Graph 检查，含 96 次变更输入重放、48 个错误对照，全部通过 |

覆盖 FP16/BF16、2D/4D、dim=1～256 二次幂及合法线程/布局。首次负例工具没有捕获非法布局产生的 ValueError，修正工具后完整通过；初始错误、修正和原始记录均保留，未更改 C++ 实现来解决该工具问题。

## 三轮普通调用配对

基线 `bce004ec`。在 RTX 4090 D / NGC Torch 2.6 / CUDA 12.8 上，目标 rows=1/17/256、dim=8/64/256、FP16/BF16；另有 rows=65536 对照。72 个配置 × 三轮共 216 组，每组九次交替测量三种路径（基线分配、候选分配、候选 out）。每个计时样本调用 2000 次，前后 CUDA 同步。

| 接口 | 三轮目标配置几何平均加速比 |
|---|---|
| Hadamard | 1.747 / 1.756 / 1.751× |
| Hadamard + INT4 | 2.008 / 2.023 / 2.006× |
| INT4 | 2.030 / 2.046 / 2.035× |

每方法每轮要求至少快 10%，任何配置或原接口回退不超过 5%，全部满足；原接口最大观测回退 2.765%以内。计时包含普通 Python 调用、旧接口输出分配/返回处理和 GPU 完成；out 的一次性预分配在计时之外。该收益要求复用缓冲区，不等于内核加速、初次分配收益或模型端到端提升。

[固定原始证据](https://github.com/a962695448-rgb/Learning-CUDA/tree/008d11dff2aa54d3bc02a3c54bdac78667daf882/03_hadamard_tc/a962695448-rgb/results/out-buffers-20260917)包含完整计时、协议、校验、源码指纹、错误记录和准备脚本。单独正确性可执行 `python scripts/verify_out_buffers.py --build-directory /tmp/out-build --json /tmp/new-out-report.json`。本轮仅在上述 NVIDIA 单卡环境实测。
