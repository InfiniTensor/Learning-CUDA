# 第八阶段证据索引

- `source.sha256`：CMake、稳定性测试、远端测试二进制、远端覆盖摘要和本地
  `coverage.csv` 的校验和，以及构建/CTest/Sanitizer 状态。
- `../../../results/stage8_stability/coverage.csv`：覆盖类别、组合数和结果摘要。
- `../../../docs/STABILITY_MATRIX.md`：Exact/IVF 实现选择、适用边界和安全回退。

远端长任务全部使用 nohup：`stage8-stability-build-20260910` 和
`stage8-stability-sanitizer-20260910`。日志与逐任务状态保存在
`/data/purplepulse-work/`；仓库不复制 sanitizer 大日志。
