# 摩尔线程第 2 次优化：算术瓶颈与诊断

性能表见 [优化日志](../../docs/MUSA_OPTIMIZATION_LOG.md)。平台仍为 S4000 / MUSA 5.1.0 / mp_22，2026-09-15。正式基准固定 FP32、block+nearest，3 次预热、20 次测量。

| 文件/目录 | 用途 |
|---|---|
| `after-divide/benchmark/` | 完整 1M/4M/16M 基准、原始 JSONL、环境和测量时源码哈希 |
| `recheck/`、`recheck.py` | 本轮修改前后交替复测，保留二进制 SHA256；优化日志使用此组数据 |
| `arithmetic_bench.cu`、`arithmetic-candidates.log` | 4M 算术隔离测试，不是正式 pipeline 性能 |
| `stage-before-*.jsonl`、`stage-after-*.jsonl` | 分阶段 event 时间，使用 `Core/backends/musa/tools/stage_profile.cu` |
| `profile-mxfp8.log`、`before-mxfp8.tsv` | MUPTI 失败证据；0 条记录，不是有效 profile |
| `tool-inventory.log` | 已安装工具/设备的只读检查及编译器 sanitizer 帮助 |
| `final-ctest.log`、`final-ctest-details.log` | MUSA 10/10 测试，含 1,048,576 对算术样本及双侧输出哨兵 |
| `cuda-ctest.log` | CUDA 9/9 回归；CUDA 除法实现未改 |
| `evaluation/summary.json`、`evaluation-summary.json` | 144 组误差/压缩/CPU 校验记录，含输入哈希 |
| `before/Core-source.tar.gz` | 修改前 Core 源码快照；远程同目录另外留存二进制，均不提交 Git |
| `SOURCE_SHA256SUMS.txt`、`SHA256SUMS.txt` | 最终 Core 源码及本目录证据完整性清单 |

隔离测试 mode：0=复制输入；1=修改前 `__fdiv_rn`；2=普通 FP32 除法；3=FP64 乘积转 FP32；4=普通 FP32 RN 乘法；5=FP64 商转 FP32。日志产生于修改前；脚本 mode=1 已显式固定为旧 intrinsic，避免公共 helper 更新后失去对照含义。普通 FP32 除法仅作性能诊断，不能用于正式数值路径。

```bash
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_22 -std=c++17 -O3 -ffp-contract=off \
  -DLP_BACKEND_MUSA -ICore/backends/musa/include -ICore records/musa-opt02/arithmetic_bench.cu \
  -L/usr/local/musa/lib -lmusart -Wl,-rpath,/usr/local/musa/lib -o Core/build-musa/arithmetic_bench
Core/build-musa/arithmetic_bench
```

最终生产变更仅为 MUSA 的 `divide_rn` 替代实现，保留原乘法及所有编码/scale/舍入规则。测试扩展不参与性能计时；GPU 输出前后保护区只能发现覆盖用例的越界写，不能证明没有越界读/数据竞争。

MUPTI 当前报 MT-Perf 无法建立硬件连接，随后进程异常退出。插件仅编译及尝试过，未验证成功采集；本轮没有使用无效 trace 得出硬件利用率结论。需要平台方核对宿主机 MT-Perf、容器权限/设备映射，并提供 MUSA 专用 memcheck/racecheck 工具。使用方式见 [诊断工具说明](../../Core/backends/musa/tools/README.md)。
