# v6 方向6：Shared bank conflict

基础版本：`v3_vectorized_memory`。本版本按“正收益即晋级”规则晋级。

NVFP4 block原先用`uint8_t c[16]`保存各线程的E2M1 code，多个线程会落到同一个32-bit shared-memory bank。本版本改为`uint32_t c[16]`，使连续lane映射到不同bank，数值和最终8-bit打包格式不变。

- NVFP4 block量化：v3 8.884736 ms，v6 8.882688 ms，吞吐提升约0.023%。
- 预热10次、计时50次、交错7轮取median中位数。
- `weights.luxq`及反量化输出与v3逐字节一致。
- 其余kernel源码与v3相同。

收益很小，但符合用户指定的无固定百分比门槛规则，因此方向7基于v6继续。
