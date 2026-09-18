# 摩尔线程优化日志

同一格式单独计算：相对上一轮 = 上一轮保存的 median / 当前 median；相对基线 = 第0轮优化前的 median / 当前 median。基线固定使用 records/musa-opt01/baseline，不使用当轮 before 复测替换。

历史条件：S4000、MUSA 5.1.0、mp_22；FP32、block/nearest；1M/4M/16M，预热3次、测量20次；量化完整 `resident_gpu` 序列，单位ms。旧记录没有实际输入哈希，仅作同配置历史趋势；新轮次遵守 [固定测试协议](BENCHMARK_PROTOCOL.md)。

## MXFP8

| 轮次 | 元素数 | median ms | P95 ms | 相对上一轮 | 相对基线 |
|---|---:|---:|---:|---:|---:|
| 第0轮：优化前基线 | 1M | 0.676629 | 0.681691 | — | 1.00x |
| 第0轮：优化前基线 | 4M | 2.564229 | 2.564892 | — | 1.00x |
| 第0轮：优化前基线 | 16M | 10.094629 | 10.097554 | — | 1.00x |
| 第1轮：shuffle+向量化 | 1M | 0.505040 | 0.505349 | 1.34x | 1.34x |
| 第1轮：shuffle+向量化 | 4M | 1.624251 | 1.624937 | 1.58x | 1.58x |
| 第1轮：shuffle+向量化 | 16M | 6.327497 | 6.332503 | 1.60x | 1.60x |
| 第2轮：严格除法替代 | 1M | 0.166549 | 0.167406 | 3.03x | 4.06x |
| 第2轮：严格除法替代 | 4M | 0.399097 | 0.399886 | 4.07x | 6.43x |
| 第2轮：严格除法替代 | 16M | 1.360480 | 1.363383 | 4.65x | 7.42x |

数据来源（按轮次）：

- [优化前基线](../records/musa-opt01/baseline/summary.json)。
- [shuffle+向量化](../records/musa-opt01/final-benchmark/summary.json)。
- [严格除法替代](../records/musa-opt02/recheck/summary.json)，只取 stage=after-divide。

## NVFP4

| 轮次 | 元素数 | median ms | P95 ms | 相对上一轮 | 相对基线 |
|---|---:|---:|---:|---:|---:|
| 第0轮：优化前基线 | 1M | 0.622149 | 0.627771 | — | 1.00x |
| 第0轮：优化前基线 | 4M | 2.090983 | 2.096137 | — | 1.00x |
| 第0轮：优化前基线 | 16M | 8.095337 | 8.100206 | — | 1.00x |
| 第1轮：shuffle+向量化 | 1M | 0.597931 | 0.603360 | 1.04x | 1.04x |
| 第1轮：shuffle+向量化 | 4M | 1.747714 | 1.751543 | 1.20x | 1.20x |
| 第1轮：shuffle+向量化 | 16M | 6.718057 | 6.720297 | 1.21x | 1.21x |
| 第2轮：严格除法替代 | 1M | 0.238274 | 0.243680 | 2.51x | 2.61x |
| 第2轮：严格除法替代 | 4M | 0.495760 | 0.498514 | 3.53x | 4.22x |
| 第2轮：严格除法替代 | 16M | 1.647989 | 1.652571 | 4.08x | 4.91x |

数据来源（按轮次）：

- [优化前基线](../records/musa-opt01/baseline/summary.json)。
- [shuffle+向量化](../records/musa-opt01/final-benchmark/summary.json)。
- [严格除法替代](../records/musa-opt02/recheck/summary.json)，只取 stage=after-divide。

## 改动与验证

- 第1轮：shuffle 归约 + float4 加载/合并写出；纯 shuffle 广播未通过测试，保留 shared scale 广播。子步骤原始测量保留在 records/musa-opt01，不另造加速链。
- 第2轮：用 FP64 商一次 RN 转回 FP32 替代高开销 __fdiv_rn，CUDA 路径未改。百万对算术测试、144组 CPU 评估、MUSA 10/10 与 CUDA 9/9 CTest通过；这些是正确性验证，不是性能数据。
- MUPTI 采集报 MT-Perf 硬件连接失败；完整设备内存/竞争检查仍缺工具，不能声明通过。

[诊断步骤](../Core/backends/musa/tools/README.md)；[前版日志快照](../records/benchmark-audit/MUSA_LOG_before_cleanup.md)。历史原始 JSON 保留，不覆盖；表格由脚本统一计算，不手填加速比。

## 各版本相对基线的加速比

基线为第0轮，同格式、同元素数的 `quant_optimized / resident_gpu` median 相除。

### MXFP8

| 版本 | 1M | 4M | 16M |
|---|---:|---:|---:|
| 第0轮：基线 | 1.00x | 1.00x | 1.00x |
| 第1轮：shuffle+向量化 | 1.34x | 1.58x | 1.60x |
| 第2轮：严格除法替代 | 4.06x | 6.43x | 7.42x |

### NVFP4

| 版本 | 1M | 4M | 16M |
|---|---:|---:|---:|
| 第0轮：基线 | 1.00x | 1.00x | 1.00x |
| 第1轮：shuffle+向量化 | 1.04x | 1.20x | 1.21x |
| 第2轮：严格除法替代 | 2.61x | 4.22x | 4.91x |
