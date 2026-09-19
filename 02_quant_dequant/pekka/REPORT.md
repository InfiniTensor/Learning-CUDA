# MXFP8 / NVFP4 CUDA 量化项目实现报告

## 1. 结论与验收

本项目实现了完整流水线：QDAT（FP16/FP32 输入）→ CUDA scale/quantize/pack → 自描述
QLOW → QLOW reload/checksum → CUDA unpack/dequantize → FP16/BF16/FP32 QDAT 与 JSON 指标。
实现只使用普通 CUDA C++，不依赖原生 FP8/FP4 intrinsic、Tensor Core 或第三方量化库。

本次 GPU 为 WSL2 的 NVIDIA GeForce RTX 5060 Ti，compute capability 12.0，CUDA 13.3。
最终 CTest 为 5/5 通过；七个长度 `1,15,16,17,31,32,33` 的两种 block 格式共 14 条
流水线通过 CPU/GPU 逐字节量化对拍和 FP32 反量化对拍。

## 2. 第一版规格决定

|项目|决定|
|---|---|
|MXFP8|OCP E4M3FN，最大有限值 448，每 32 元素一个 E8M0 scale|
|NVFP4|E2M1（`0,.5,1,1.5,2,3,4,6`），每 16 元素一个 E4M3 local scale + FP32 tensor scale|
|scale mode|block 或 tensor；tensor 对整张矩阵只生成一组 local scale|
|舍入|nearest ties-to-even，或使用 seed+元素 index 的可复现 stochastic rounding|
|非有限|amax 忽略 NaN/Inf；NaN→0；Inf 饱和到同符号最大有限值|
|尾块|只扫描有效元素；奇数 FP4 最后 byte 的高 nibble 清零|

全零组令 scale=1。误差在恢复为 FP32 后统计，非有限输入单独计数、不参与 MAE/MSE。

## 3. 数学与位布局

E4M3 为 `s eeee mmm`、bias=7；`e=0` 是 subnormal，`e=15,m<=6` 为有限数，`0x7e=448`，
`0x7f/0xff` 是 NaN。E2M1 为 `s ee m`、bias=1，两个 nibble 共用一个 byte：偶数元素在低
四位，奇数元素在高四位。encoder 枚举有限候选并选择最近值，中点选择偶数最低位。

MXFP8 对 block `B`：

```text
a_max=max(abs(x_i))     s_ideal=a_max/448
s=smallest E8M0 power >= s_ideal
q_i=E4M3_encode(x_i/s)  x_hat_i=E4M3_decode(q_i)*s
```

NVFP4：

```text
g=tensor_amax/(6*448)   l_B=E4M3_encode(block_amax/(6*g))
q_i=E2M1_encode(x_i/(g*decode(l_B)))
x_hat_i=decode(q_i)*decode(l_B)*g
```

GPU 上 NVFP4 tensor amax 使用 grid-stride 扫描和非负 float 的 `atomicMax`。这使 1024²
量化从优化前约 50 ms 降至 0.072–0.101 ms，且误差完全不变。

## 4. 文件协议

QDAT 是 24-byte header：`QDAT`、version、dtype（1=FP16、2=FP32、3=BF16）、reserved、
rows、cols，随后是严格匹配 shape 的 row-major payload。

QLOW v1 header 固定 92 bytes，记录 magic/version、format、original dtype、scale mode、
rounding、shape、block size、packed bytes、scale bytes、三个 section offset、FNV-1a checksum
和 total bytes。packed data、local scales、global FP32 scale 各从 16-byte 边界开始。reader
检查枚举、shape overflow、期望 packed/scale 数量、section 顺序/对齐/边界、checksum 和正的
有限 global scale；这使小型实验也不会静默接受损坏文件。

## 5. 逐文件、逐函数说明

### 公共头文件

`include/qd/common.hpp` 定义 `DataType/Format/ScaleMode/Rounding`，以及 `Config`、`Matrix`、
`QuantizedData`、`Metrics`、`Timings`；`element_count` 计算元素数，`data_type_size` 返回
payload 宽度，`to_string` 函数供 JSON/CLI 使用。`config.hpp`、`io.hpp`、`reference.hpp`、
`metrics.hpp`、`cuda_ops.hpp` 分别声明配置、I/O、CPU reference、指标和 GPU API。

`include/qd/formats.cuh` 是 CPU/CUDA 共用规则：`decode_e4m3/decode_e2m1/decode_e8m0` 解码；
`encode_e4m3/encode_e2m1` 处理饱和、NaN 和 ties-to-even；两个 stochastic encoder 在相邻
可表示值之间按距离比例选择；`encode_e8m0_ceil` 选择不小于理想 scale 的二次幂；`is_nan`
是 host/device NaN 判定；`uniform01` 是 seed+index counter hash。

### CPU、I/O 与 CLI

`src/config.cpp` 的 `trim/unquote` 处理简单 TOML 文本，`read_config` 转换字段并报告错误，
`validate_config` 固定 block size 并校验 rounding 枚举。`src/io.cpp` 的 `half_to_float` 显式
处理 FP16 normal/subnormal/Inf/NaN；`read_qdat/write_qdat` 负责 QDAT；`align16`、
`fnv1a`、`write_padding` 是 QLOW 辅助；`qlow_file_size` 计算真实文件大小；`write_qlow`
写 header/sections，`read_qlow` 执行完整协议验证。

`src/reference.cpp` 中 `group_count/group_range/scale_index` 管理 tensor、block 和尾块，
`finite_amax` 忽略非有限数，`quantize_reference/dequantize_reference` 是慢但透明的 CPU
真值路径。`src/metrics.cpp` 的 `calculate_metrics` 求 max abs、MAE、MSE 和非有限计数；
`write_metrics_json` 写压缩率、kernel 时间与 logical effective GB/s。

`src/main.cpp` 的 `Arguments/parse_arguments/print_usage` 处理 CLI，`check_reference` 要求
CPU/GPU packed bytes、scales、global scale 相同；`main` 串起 config→input→warmup→quantize→
reference→QLOW write/reload→dequantize→QDAT/JSON，任何异常都返回非零码。

### CUDA

`src/cuda_ops.cu` 中 `check_cuda` 统一错误处理，`DeviceBuffer` 和 `Event` 负责 RAII 资源。
`finite_amax_range` 扫描 scale 组；`mxfp8_scale_kernel`、`mxfp8_quantize_kernel` 是 FP8 两阶段；
`tensor_amax_kernel`、`nvfp4_global_scale_kernel`、`nvfp4_local_scale_kernel` 完成 NVFP4 两级
scale；`nvfp4_quantize_kernel` 一线程独占一个 packed byte；`dequantize_kernel` 解包并同时写
FP32、FP16、BF16 或 FP32 raw output。`elapsed_average` 只统计 CUDA event kernel 区间，
`groups_for` 计算 scale 数；`quantize_gpu/dequantize_gpu` 完成分配、H2D/D2H、迭代和结果封装。

`src/cuda_smoke.cu` 的 `check_cuda`、`smoke_kernel`、`main` 只验证设备、launch、copy/free，
没有混入量化逻辑；无 GPU 返回 77，CTest 显示为 Skipped。

### 测试、脚本和工具

`tests/test_formats.cpp` 穷举 256 个 E4M3 pattern、16 个 E2M1 pattern 及边界；
`tests/test_reference.cpp` 验证 odd nibble、tensor scale、尾块和全零；`tests/test_io.cpp`
验证 FP16 最小 subnormal `2^-24` 与 QLOW round-trip；`tests/run_pipeline_test.py` 先运行
smoke，再覆盖 MXFP8/NVFP4 的 block/tensor 与 FP16/BF16/FP32 输出。

`tools/generate_test_data.py` 的 `write_qdat/inspect_qdat/make_values/generate_one/parse_args/main`
分别写入、检查、生成 uniform/normal/outlier、组织单文件/套件和命令分发；`tools/summarize_results.py`
读取 JSON 并输出 CSV。`scripts/check_env.sh` 探测 WSL、nvcc、GPU、编译器与分析工具；
`bootstrap.sh` 完成冷启动构建/CTest/测试数据；`run_correctness.sh` 覆盖七个非对齐尺寸；
`run_benchmarks.sh` 对三类 1024² 数据各跑两种格式（10 warmup、100 iterations）。

## 6. 实验结果

|format|data|max abs|MAE|MSE|quant ms|dequant ms|file compression|
|---|---|---:|---:|---:|---:|---:|---:|
|MXFP8|uniform|0.2500|0.06936|0.008907|0.6101|0.05558|3.878x|
|MXFP8|normal|0.2447|0.01801|0.0007081|0.5967|0.03142|3.878x|
|MXFP8|outlier|1.457e18|1.390e12|2.025e30|0.5972|0.02421|3.878x|
|NVFP4|uniform|1.0000|0.2656|0.1239|0.07198|0.02996|7.110x|
|NVFP4|normal|0.6362|0.07142|0.009040|0.07390|0.03262|7.110x|
|NVFP4|outlier|8.796e12|8.389e6|7.379e19|0.1011|0.02405|7.110x|

记录位于 `results/benchmarks/*.json`，汇总在 `results/benchmarks/summary.csv`。outlier 的
`1e20` 会支配所在组 scale，故绝对误差很大；normal 数据显示 MXFP8 精度明显更好。NVFP4
更高压缩率来自两元素一 byte，但第一版 MXFP8 encoder 枚举候选、NVFP4 只枚举 7 个正值，
因此该软件 baseline 的速度不能代表原生硬件格式吞吐。

## 7. 限制与后续

当前只实现 E4M3FN、不实现 E5M2；benchmark 报告多次平均而非 p10/median/p90；尚未提交
正式 ncu/nsys/compute-sanitizer profile。后续推荐顺序是：先用更多 golden vectors 扩充
stochastic 覆盖，再把 E4M3 枚举 encoder 换成常数时间位运算并保留
exhaustive tests，最后根据 profiler 证据做 warp reduction、vectorized load、kernel fusion。
QLOW 的 FNV-1a 只用于损坏检测，不提供密码学完整性；大 shape 仍一次性分配 host/device buffer。

## 8. 复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/run_correctness.sh
./scripts/run_benchmarks.sh
```

固定 seed 为 `20260827`。`.qdat/.qlow` 是可重生成的大文件，已由 `.gitignore` 排除；JSON/CSV
结果和本报告保留，保证代码、协议、reference、GPU kernel、测试和结论之间可审计。
