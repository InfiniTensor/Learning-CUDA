# 本次验证记录

本页原表为 2026-09-08 的 v1 历史结果。2026-09-09 的 v2 扩展、优化和完整验收记录
见 [FINAL_REPORT.md](../FINAL_REPORT.md)，性能原始记录见 `../records/_archive/2026-09-09-legacy-cli/performance-warp-20260909/`。

日期：2026-09-08。GPU：RTX 3060 Laptop，驱动 596.08。
编译器：CUDA 12.0.140，目标 sm_86，Release；GPU 检查工具来自 CUDA 12.9.79 独立包。

| 检查 | MXFP8 | NVFP4 |
|---|---|---|
| CTest | 27/27 PASS | 27/27 PASS |
| 冻结 golden 用例 | 5/5 PASS | 5/5 PASS |
| 新增量化输入 | 139 PASS | 141 PASS |
| 编码舍入样本 | 800 PASS（E4M3） | 800 PASS（E4M3、E2M1） |
| 反量化输出 | FP32/FP16/BF16 PASS | FP32/FP16/BF16 PASS |
| CPU 输入与 golden SHA256 | 15/15 OK | 15/15 OK |
| memcheck，padding=32 | 0 errors，0 bytes leaked | 0 errors，0 bytes leaked |
| racecheck | 0 errors，0 warnings | 0 errors，0 warnings |
| synccheck | 0 errors | 0 errors |

回归用例包括长度 0～65 的普通输入和全负零输入、255/256/257/1023/1024/1025
线程边界、正 scale 下的负零、零 scale 下的正零，以及 NVFP4 的 block scale 舍入为零。
E2M1/E4M3 中点测试同时覆盖中点和它相邻的 FP32 值，使用 CPU reference 的候选选择规则。

16 位输出还通过独立 oracle 验证 nearest-even 中点、subnormal、下溢、溢出、
符号零和非有限值；文件头、payload 和长度检查通过。人为破坏输出的反向测试被正确拒绝。

原始日志位于各模块 `build/tests/results/ctest.log`、`cpu_hashes.log`、
`memcheck.log`、`racecheck.log`、`synccheck.log`。完整 CTest 输出在 `build/Testing/Temporary/LastTest.log`。
复现命令见 [README.md](README.md)。本次仅验证正确性，不作性能结论。

## 正式性能基准

基准程序为 `benchmark`，每个规模先预热 3 次，再重复 20 次，报告 wall-clock 中位数和 P95。计时包含当前 API 每次调用的设备内存申请、释放以及必要的同步，因此是端到端调用成本；有效带宽按输入、输出和 scale/payload 字节数估算，适合比较同一工程内的趋势，不等同于显存理论带宽。

测试规模为 1M、4M、16M elements，FP32 输入，RTX 3060 Laptop（sm_86）：

| 格式 | 操作 | 1M median/P95 ms | 4M median/P95 ms | 16M median/P95 ms |
|---|---|---:|---:|---:|
| MXFP8 | quantize | 2.3227 / 2.6699 | 6.5705 / 7.1939 | 23.5946 / 25.2928 |
| MXFP8 | dequantize FP32 | 1.9578 / 2.4698 | 5.1083 / 5.9712 | 38.3431 / 41.7978 |
| NVFP4 | quantize | 4.3750 / 4.7366 | 13.5439 / 14.2929 | 49.0400 / 51.1727 |
| NVFP4 | dequantize FP32 | 1.9608 / 2.1517 | 4.9452 / 6.0616 | 38.7758 / 42.5301 |

对应中位数有效带宽（GB/s）依次为：MXFP8 2.271/3.212/3.578（量化）、2.695/4.131/2.201（反量化）；NVFP4 1.094/1.413/1.561（量化）、2.440/3.870/1.974（反量化）。NVFP4 量化较慢主要来自 global-max、拷回和 global-scale 计算的端到端步骤；后续若追求吞吐，应将该步骤改为设备端归约并复用分配。

原始基准日志位于 `CUDAMXFP8/build/tests/results/benchmark.log` 和 `CUDANVFP4/build/tests/results/benchmark.log`。本次优化范围保持 golden 字节级兼容，未牺牲正确性；27/27 CTest 和 sanitizer 结果仍以本文件前表为准。
