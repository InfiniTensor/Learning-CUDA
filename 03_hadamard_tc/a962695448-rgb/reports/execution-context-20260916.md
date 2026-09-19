# 非默认流与 CUDA Graph 补充验收

2026-09-16 在 RTX 4090 D 24GB 单卡实测。新增测试脚本，原 51 份计算/构建源码保持不变。

| 检查 | 结果 |
|---|---|
| 接口及配置 | 120 组通过；hadamard / hadamard_int4 / quantize_int4，FP16/BF16、128/256 线程、2D/4D |
| 非默认流 | 120 次延迟生产者与消费者正确性检查通过 |
| CUDA Graph | 每组四次变更输入后的重放，共 480 次通过 |
| 负例 | 120 个默认流消费者错误、120 个旧输出错误，全部检出 |
| 参考及输入 | NumPy 独立 FWHT/INT4 参考逐位一致，输入未被改写 |

维度覆盖 1/8/64/256，布局覆盖合法的 original/auto/packed/contiguous256 组合。错误对照不能被检出时测试失败，避免将缺少区分能力的测试写为通过。负例、重放次数不累计为更多独立配置。

从本项目目录执行：

```bash
export PATH=/usr/local/cuda/bin:$PATH
python scripts/verify_execution_context.py --build-directory /tmp/hadamard-context-build --json /tmp/new-context-report.json
```

输出路径必须不存在。延迟生产者测试要求 `torch.cuda._sleep` 可用；未提供或负例失效时明确失败。CUDA Graph 预热、流依赖和固定内存更新规则见 [PyTorch 2.6 官方说明](https://docs.pytorch.org/docs/2.6/notes/cuda.html)。

这是正确性补测，未修改内核、不宣称新的加速比。单卡代表性配置不能代替任意流调度、多 GPU 或新 A100 验证。实际环境为 NGC Torch 2.6.0a0+ecf3bae40a.nv25.01 / CUDA 12.8 / NumPy 1.26.4。

[固定原始证据](https://github.com/a962695448-rgb/Learning-CUDA/tree/3c0db6daa462bfba7ec0904e65d5a841064ab5b3/03_hadamard_tc/a962695448-rgb/results/execution-context-20260916)包含每组配置、每次重放、负例记录、原始日志、脚本和指纹清单。结果回收后逐文件 SHA-256 核对通过。
