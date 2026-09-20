# GPU 向量检索引擎总结报告

## 1. 系统范围

系统实现 squared-L2、FP32/FP16 文件输入、CPU 精确参考、GPU Exact、IVF-Flat、
索引持久化、批量查询、K=1/10/50/100、性能日志与质量日志。当前只适配 NVIDIA CUDA；
inner product、cosine、IVF-PQ 和国产 GPU 不在本版范围。

## 2. 精确检索

输入文件为 row-major。GPU 内部转为 dimension-major `[D,N]`。距离 kernel 中一个线程
处理候选向量；同一 warp 在固定维度读取连续候选，形成合并访问。query 由 global memory
加载到 shared memory。距离用 FP32 FMA 累加：

```text
d(q,x) = sum_j (q_j - x_j)^2
```

不计算平方根。它不改变最近邻次序，也避免额外指令和数值扰动。

## 3. Top-K

若保存 `[batch,N]` 距离矩阵，`N=10^6,batch=128` 已需 512 MB。系统改用分层归并：

```text
1024 candidates/block -> block radix Top-K -> recursive radix merge -> K
```

排序键为 `uint64(distance_bits)<<32 | original_id`。L2 非负 FP32 的 bit pattern 与数值顺序
一致；低 32 bit 固定 tie-break。无效候选和非有限距离使用最大 key。

该结构同时服务 Exact 的跨候选块归并，以及 IVF 的跨倒排桶归并。避免重复实现两套排序语义。

## 4. IVF-Flat

训练从数据库确定性抽样。初始中心来自均匀分布的训练样本。每轮 Lloyd 迭代：GPU Exact
搜索产生最近中心 ID，CPU 以 double 累加各中心和并更新 FP32 centroid。空中心保留上一轮值。
随后 GPU 对全库执行 Top-1 assignment，Host 计算桶 offsets，并把向量重排为全局
dimension-major 布局；每个桶在 position 维连续。

查询分两级：GPU 搜索 `nprobe` 个中心；每个 `(query,probe)` block 顺序扫描一个桶，
不断把当前 K 个最佳值与新候选合并；最终再跨桶归并。输出始终映射回原始向量 ID。

## 5. 正确性

CPU 参考以 `(distance,id)` 排序。GPU Exact 的 ID 必须完全一致，score 使用浮点容差。
测试包含重复向量，确保 tie 规则生效。

IVF 的强不变量为：

```text
nprobe = nlist => IVF IDs == Exact IDs
```

Recall 定义：

```text
recall@K = sum_q |ANN(q) intersect Exact(q)| / (Q*K)
```

平均距离误差为排序后各 rank 的非负超额 L2 距离均值。

## 6. 性能口径

- `total_ms`：search API wall time，包含查询/结果传输和分配，不含文件 I/O及索引加载。
- `gpu_ms`：CUDA event 覆盖批次 H2D、kernel 与结果 D2H。
- `QPS = num_queries / total_ms`。
- P50/P99：各 batch 的 wall latency；正式 P99 至少采集 100 个 batch 样本。
- `peak_device_bytes`：索引与显式 workspace 的请求字节数，不冒充 profiler 实测峰值。
- CPU 加速比：同一数据库、查询和 K 的 CPU wall time / GPU total time。

## 7. 当前验证

开发环境：RTX 5060 8 GiB、CUDA 13.2、SM120、WSL2。

已完成自动化验证：

- CPU Exact 对 GPU Exact；
- K=1、10、50、100；
- N=513 与 batch=7 的尾部；
- 重复向量 tie；
- FP16/FP32 文件；
- IVF build/save/load；
- `nprobe=nlist` 与 Exact 等价；
- CLI 端到端性能/质量日志。

开发自测只用于正确性，不作为最终性能数字。

### 7.1 百万规模实测

数据：`N=1,000,000`、`D=128`、`Q=1000`、FP32、K=10；生成器产生 64 个高斯簇。
GPU 当时被其他进程占用约 6.5 GiB，存在明显抢占，因此这些数字证明规模能力和设计趋势，
不作为独占 GPU 峰值。

| 模式 | batch | nlist/nprobe | 总时间 | QPS | P50 batch | P99 batch | 显式显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact + CPU 全量核对 | 128 | - | 5320.2 ms | 188.0 | 241.9 ms | 2167.6 ms | 522.2 MB |
| Exact latency run | 10 | - | 8018.0 ms | 124.7 | 14.8 ms | 1576.3 ms | 512.8 MB |
| IVF-Flat | 128 | 4096/16 | 171.1 ms | 5844.1 | 16.4 ms | 35.2 ms | 518.4 MB |

IVF 建索引时间 26.77 s；`recall@10=0.8696`，平均超额 squared-L2 距离 0.05320。
CPU Exact 同规模耗时 70.67 s；GPU Exact Top-10 ID 全部一致，加速 13.28 倍。
相对已核对的 batch=128 GPU Exact wall time，IVF 查询快 31.1 倍。Exact P99 中存在秒级 GPU
抢占离群值；batch=10 共 100 个延迟样本，但仍受共享 GPU 污染，不能解释为 kernel 本身尾延迟。
原始 JSON 位于 `results/million/`。

### 7.2 nprobe / batch 开发 sweep

在 `N=100,000,D=128,Q=100,nlist=256,K=10` 上，nprobe 从 1/4/8 增加时，
recall 从 0.550/0.948/1.000 上升，平均距离误差从 0.36550/0.02586/0 降低。
小样本且 GPU 非独占，QPS 不严格单调；完整原始日志位于 `results/dev_100k/`，不以其中
单次峰值作为最终性能结论。

### 7.3 profiler

nsys 在 `N=65536,D=128,Q=128,batch=16,K=10` 上成功采集。GPU kernel 时间分布：

- `distance_partial_topk_kernel`：5.811 ms / 8 次，92.4%；
- 数据转置：0.398 ms / 1 次，6.3%；
- `merge_topk_kernel`：0.078 ms / 8 次，1.2%。

结论：归并不是当前主瓶颈；距离读取和局部 radix Top-K 是优化中心。CUDA API trace 还显示
首次 `cudaMalloc`/`cudaFree` 成本很高，生产服务应复用 index 和 workspace。

ncu 启动成功，但 WSL/WDDM 返回 `ERR_NVGPUCTRPERM`，当前用户无权访问硬件性能计数器。
需要在 Windows 宿主机管理员权限下启用 NVIDIA debugger/performance-counter interface，或在
原生 Linux 调整 `NVreg_RestrictProfilingToAdminUsers` 后重跑。该失败未伪装成 ncu 数据。

## 8. 预期瓶颈与 profiling 目标

Exact 的主要流量来自数据库向量读取，预期受 DRAM bandwidth 限制；K 增大后 radix sort、
局部结果写回和 merge 比例上升。ncu 重点观察 DRAM throughput、global load efficiency、
occupancy、register pressure 和 sort 指令占比。

IVF 访问量约为 `nprobe*N/nlist`，但桶长不均会造成 block 负载失衡。nsys 重点查看短 kernel、
批次间空洞、H2D/D2H 串行和中心搜索开销。

## 9. 后续方向

1. 一个 block 同时处理多个 query，复用数据库读取。
2. 用 GPU reduction 更新 centroid，消除训练 assignment 的 D2H。
3. 候选 tile 化，进一步限制十亿级库的 Top-K workspace。
4. FP16 索引常驻、FP32 累加，降低一半显存和带宽。
5. 增加 inner product/cosine、IVF-PQ 和国产平台后端。
