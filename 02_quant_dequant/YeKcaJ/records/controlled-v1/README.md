# 固定输入基线 v1

2026-09-15 对当前保留实现重新测量：CUDA / RTX 3060 Laptop 与 MUSA / MTT S4000。
本目录是新协议基线，不是六个历史版本的重测。

- 条件：[固定协议](../../docs/BENCHMARK_PROTOCOL.md)。三个规模的实际 FP32 文件 SHA256 两端完全一致。
- 性能：[CUDA](cuda/RESULTS.md)、[MUSA](musa/RESULTS.md)，各自 environment.json 保存输入、源码、二进制哈希和环境；JSONL 保存程序输出。
- 跨平台原始结果分别保存在 `cuda/` 和 `musa/`；跨平台耗时比不算作优化轮次加速比。
- 验证：CUDA CTest 9/9 通过，MUSA CTest 10/10 通过，日志为 ctest-cuda.log / ctest-musa.log。

计时后仅补充了 Python 输入校验与文档，未修改被测 kernel。保留运行时 environment.json 原始哈希，不用后来的工具文件哈希覆盖测量来源。
