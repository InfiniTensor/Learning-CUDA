# 第 3 次优化验证

- `source-before.tar.gz`：修改前完整 Core 源码（含当时用户的未提交配置及输入输出路径修改），不含 build/cache。
- `before/benchmark` 与 `after/benchmark`：同环境、同一 benchmark、预热3次/20次计时，1M/4M/16M，FP32 block+nearest；各自 environment.json 记录源码 SHA256。
- `before/profile` 与 `after/profile`：同一 nsys 4M 采集，完整统计包含内部对照与默认路径；比较单 kernel 中位数，不比较累计占比。
- `repeat`：先新后旧运行 NVFP4，各规模20次复测；旧可执行文件为修改前保留版本。16M 未见稳定收益，P95 波动保留在原始数据中。
- `validation/LastTest.log`：8项 CTest 全部通过。`core_reduction` 含220组，逐块及全局最大值与独立 CPU 扫描比较，并检查 global_scale 位模式。
- `validation/*-memcheck.log`、`*-racecheck.log`、`*-synccheck.log`：归约专项、MXFP8自测、NVFP4自测均通过，9份日志；无错误/数据竞争/同步错误，memcheck 无泄漏。

本轮使用 NVIDIA 官方 CUDA Sanitizer 12.9.79 发行包（工具报告版本2025.2.1），避免系统旧工具与驱动不兼容。下载工具仅放临时目录，没有替换系统 CUDA。
测试另修复了已有文件测试将输出类型写死为 BF16 的断言，现在按实际配置检查文件名；没有改变输出路径实现。

复现核心检查（项目根目录）：

```bash
cmake -S Core -B Core/build -DCMAKE_BUILD_TYPE=Release
cmake --build Core/build -j 4
ctest --test-dir Core/build --output-on-failure
```

使用兼容驱动的 compute-sanitizer，依次对 `Core/build/reduction_test`、`Core/build/pipeline_nvfp4 --self-test`、`Core/build/pipeline_mxfp8 --self-test` 运行 memcheck/racecheck/synccheck。
本轮不修改 CPU reference、量化 scale 公式、nearest/stochastic 舍入或反量化输出规则。不以单次 CLI 主机耗时宣称算子加速。
