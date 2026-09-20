# Kernel versioning rules

`v0`、`v1`、`v2` 是已冻结的性能基线。不要在它们上继续调优；这样历史 CSV 才能持续比较。

新增版本流程：

1. 从最新版本复制为 `kernels/hadamard_vN.cu`；
2. 在 `KernelVersion`、`hadamard_api.cu`、CMake 的 source list 和 `benchmark/common.py` 的 `VERSION_IDS` 注册 `vN`；
3. 说明新的优化策略、dtype 行为和 workspace；
4. 运行 `./scripts/run_benchmark.sh`，把新增版本与全部旧版本、Dao 和 clone 放入同一报告。

所有版本必须计算 `x @ H(D) / sqrt(D)`；若变更数学定义，必须创建独立算子系列，不能复用该版本号。
