# v2 方向2：全局内存合并访问

基础版本：`v1_thread_block`（反量化128 threads/block）。

## 分析

逐条检查量化与反量化kernel后，输入、量化数据和输出均按相邻线程访问相邻地址：

- MXFP8量化：一个warp处理连续32个输入和输出元素。
- NVFP4量化：连续线程读取连续16个输入，偶数线程顺序写入8个packed byte。
- MXFP8反量化：连续线程读取连续code并写连续输出。
- NVFP4反量化：相邻两个线程读取同一个packed byte并写连续输出。

Nsight Compute的`MemoryWorkloadAnalysis_Tables`没有发现跨线程stride造成的地址错排，但指出窄全局加载的sector利用率不足：MXFP8平均16.5/32 byte，NVFP4平均9.0/32 byte。这里的主要来源是1-byte code加载和scale广播，不是线程地址未合并。

## 决策

本方向不强行改代码，也不产生新的晋级版本；方向3继续基于v1，通过向量化加载/存储解决窄访问问题。scale复用留给方向5处理。

Profiler原始输出保存在`output/optimization/v2_ncu_*.csv`。
