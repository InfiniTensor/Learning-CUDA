# CUDA 优化日志

保留原第0～6次编号。数值从下方链接的原始 JSON 提取；不把新重测替换成历史成绩。单位均为 ms，1M=1,048,576。

历史记录声明的条件：RTX 3060 Laptop，FP32，block/nearest，MXFP8 block=32、NVFP4 block=16，预热3次、测量20次。量化表取 `quant_optimized / resident_gpu` 的 median/P95；NVFP4 包含全局归约。第5次单列 host_api。

加速比只有两列：相对上一轮＝上一轮已保存 median ÷ 本轮 median；相对基线＝第0次同格式、同元素数的 median ÷ 本轮 median。基线固定，不随优化轮次改变。

旧实验未保存实际输入文件哈希，下面是历史记录对照；不能补称已逐字节控制输入。今后执行 [固定输入协议](docs/BENCHMARK_PROTOCOL.md)。

## 第0次：优化前基准（2026-09-10）

这是两种格式最初的参照，不能省略，也不是内部 `quant_enumeration` 的耗时。

| 格式 | 元素数 | 量化 median | 量化 P95 | FP32反量化 | FP16反量化 | BF16反量化 |
|---|---:|---:|---:|---:|---:|---:|
| MXFP8 | 1M | 0.123904 | 0.130048 | 0.031744 | 0.024576 | 0.027648 |
| MXFP8 | 4M | 0.467968 | 0.474112 | 0.111616 | 0.103424 | 0.108944 |
| MXFP8 | 16M | 1.634304 | 1.679360 | 0.404928 | 0.364544 | 0.381952 |
| NVFP4 | 1M | 0.142336 | 0.178048 | 0.025600 | 0.020272 | 0.022528 |
| NVFP4 | 4M | 0.450944 | 0.470016 | 0.088064 | 0.078848 | 0.086032 |
| NVFP4 | 16M | 1.742800 | 2.141184 | 0.339824 | 0.302080 | 0.332800 |

来源：[第0次完整数据](records/01-before/benchmark/summary.json)。

## 第1次：MXFP8 直接编码 + scale 融合（2026-09-10）

nearest 直接生成 E4M3；再将分组归约、scale 计算和编码融合。保留当时两步分别测得的数据。

上一轮＝第0次；基线＝第0次。

| 元素数 | 上一轮 median | 仅直接编码 | 本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 0.123904 | 0.055728 | 0.034816 | 0.040960 | 3.56x | 3.56x |
| 4M | 0.467968 | 0.199680 | 0.123312 | 0.128896 | 3.79x | 3.79x |
| 16M | 1.634304 | 0.708608 | 0.435648 | 0.737280 | 3.75x | 3.75x |

来源：[上一轮](records/01-before/benchmark/summary.json)、[仅直接编码](records/02-mxfp8-direct-encode/benchmark/summary.json)、[本轮最终结果](records/03-mxfp8-fused-scale/benchmark/summary.json)。

## 第2次：NVFP4 scale 直接编码 + 归约融合（2026-09-11）

block scale 使用直接 E4M3 编码，再融合分组归约与打包编码；全局最大值归约仍保留。

上一轮＝第0次 NVFP4；基线＝第0次 NVFP4。

| 元素数 | 上一轮 median | 本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |
|---:|---:|---:|---:|---:|---:|
| 1M | 0.142336 | 0.063488 | 0.064512 | 2.24x | 2.24x |
| 4M | 0.450944 | 0.142336 | 0.143360 | 3.17x | 3.17x |
| 16M | 1.742800 | 0.519136 | 0.523264 | 3.36x | 3.36x |

仅 scale 直接编码的中间步骤：旧日志只记录4M为0.238592 ms；1M/16M未单独测量。该4M数字尚未定位到独立原始 JSON，仅保留为旧日志记载，不参与正式加速比计算。

来源：[上一轮](records/01-before/benchmark/summary.json)、[本轮最终结果](records/04-nvfp4-e4m3-scale-fused/benchmark/summary.json)；中间步骤见 [原日志第2次](records/benchmark-audit/CUDA_LOG_before_cleanup.md)。

## 第3次：NVFP4 全局归约减少同步（2026-09-14）

使用 warp shuffle 归约，再合并各 warp 结果，减少整块同步与共享内存。

上一轮＝第2次；基线＝第0次 NVFP4。

| 元素数 | 上一轮 median | 本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |
|---:|---:|---:|---:|---:|---:|
| 1M | 0.063488 | 0.048128 | 0.049056 | 1.32x | 2.96x |
| 4M | 0.142336 | 0.140288 | 0.148480 | 1.01x | 3.21x |
| 16M | 0.519136 | 0.518144 | 0.534528 | 1.00x | 3.36x |

来源：[基线](records/01-before/benchmark/summary.json)、[上一轮](records/04-nvfp4-e4m3-scale-fused/benchmark/summary.json)、[本轮](records/06-warp-reduction/after/benchmark/summary.json)。当轮另测的 before 不替换上一轮保存值。

## 第4次：MXFP8 向量化加载与打包写回（2026-09-14）

float4 加载，四个 E4M3 合并为32位写回；8线程处理一个32元素分组，尾部回退标量。

上一轮＝第1次 MXFP8；基线＝第0次 MXFP8。第2、3次针对NVFP4，不改写MXFP8的参照。

| 元素数 | 上一轮 median | 本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |
|---:|---:|---:|---:|---:|---:|
| 1M | 0.034816 | 0.025600 | 0.094080 | 1.36x | 4.84x |
| 4M | 0.123312 | 0.070656 | 0.073728 | 1.75x | 6.62x |
| 16M | 0.435648 | 0.269824 | 0.279552 | 1.61x | 6.06x |

来源：[基线](records/01-before/benchmark/summary.json)、[上一轮](records/03-mxfp8-fused-scale/benchmark/summary.json)、[本轮](records/07-mxfp8-vectorized/summary.json)。

## 第5次：workspace 复用对照（2026-09-14）

本次测 host_api：同一进程每次新建/释放 workspace 与复用 workspace 对照，包含上传、量化、下载。kernel 未改。保留原实验数据，但不与 resident_gpu 基线混算加速比。

| 格式 | 元素数 | 每次新建 median | 复用 median | 复用 P95 |
|---|---:|---:|---:|---:|
| MXFP8 | 1M | 2.171398 | 1.037373 | 1.737259 |
| MXFP8 | 4M | 4.777892 | 2.792794 | 3.193150 |
| MXFP8 | 16M | 16.660836 | 11.450560 | 14.257691 |
| NVFP4 | 1M | 1.746878 | 0.963798 | 1.332427 |
| NVFP4 | 4M | 4.598268 | 2.726448 | 3.332351 |
| NVFP4 | 16M | 14.593807 | 9.878531 | 12.850389 |

来源：[同批次新建与复用数据](records/10-workspace-reuse/benchmark/summary.json)。

## 第6次：NVFP4 partial 数量调优（2026-09-14）

全局归约 partial block 上限从4096降为1024，沿用 grid-stride loop；scale 公式和编码不变。

上一轮＝第3次 NVFP4；基线＝第0次 NVFP4。采用原日志指定的 repeat 整组数据，未逐尺寸择优。

| 元素数 | 上一轮 median | 本轮 median | 本轮 P95 | 相对上一轮 | 相对基线 |
|---:|---:|---:|---:|---:|---:|
| 1M | 0.048128 | 0.041344 | 0.043008 | 1.16x | 3.44x |
| 4M | 0.140288 | 0.140224 | 0.156672 | 1.00x | 3.22x |
| 16M | 0.518144 | 0.518656 | 0.553952 | 1.00x | 3.36x |

来源：[基线](records/01-before/benchmark/summary.json)、[上一轮](records/06-warp-reduction/after/benchmark/summary.json)、[本轮](records/11-nvfp4-partials1024/repeat/summary.json)。原日志用当轮 before 算的1.18x/1.02x/1.01x，不是这里“相对上一轮保存值”的参照。

完整诊断与正确性记录见 [原日志备份](records/benchmark-audit/CUDA_LOG_before_cleanup.md)。最近的跨平台重测单独保存在 records/controlled-v1/，不插入这六次实验，也不替代上述基准。

## 各版本相对基线的加速比

基线为第0次，同格式、同元素数的 `quant_optimized / resident_gpu` median 相除。第5次只测 host_api，不混入下表。

### MXFP8

| 版本 | 1M | 4M | 16M |
|---|---:|---:|---:|
| 第0次：基线 | 1.00x | 1.00x | 1.00x |
| 第1次中间步骤：仅直接编码 | 2.22x | 2.34x | 2.31x |
| 第1次：直接编码 + scale 融合 | 3.56x | 3.79x | 3.75x |
| 第4次：向量化（最终） | 4.84x | 6.62x | 6.06x |

### NVFP4

| 版本 | 1M | 4M | 16M |
|---|---:|---:|---:|
| 第0次：基线 | 1.00x | 1.00x | 1.00x |
| 第2次：scale 编码 + 归约融合 | 2.24x | 3.17x | 3.36x |
| 第3次：归约减少同步 | 2.96x | 3.21x | 3.36x |
| 第6次：partial=1024（最终） | 3.44x | 3.22x | 3.36x |
