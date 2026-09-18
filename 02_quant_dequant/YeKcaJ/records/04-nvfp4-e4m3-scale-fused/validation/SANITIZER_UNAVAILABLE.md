# Compute Sanitizer 本轮未能运行

## 结论

第 2 轮（NVFP4）**没有** Compute Sanitizer 证据。不得声称通过。

## 复现

```
$ compute-sanitizer --tool memcheck --error-exitcode 99 ./build/pipeline_nvfp4 --self-test ../CPUNVFP4
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so
[exit 13]
```

库实际存在，但二进制未在其搜索路径中：

```
/usr/lib/nvidia-cuda-toolkit/compute-sanitizer/libsanitizer-collection.so
```

显式指定后：

```
$ compute-sanitizer --tool memcheck \
    --injection-path /usr/lib/nvidia-cuda-toolkit/compute-sanitizer \
    --target-processes all ./build/pipeline_nvfp4 --self-test ../CPUNVFP4
========= Error: Target application terminated before first instrumented API call
```

## 根因

| 项目 | 版本 |
|---|---|
| NVIDIA 驱动 | 596.08 |
| compute-sanitizer | 2022.4.1（Deb 包 nvidia-cuda-toolkit 12.0 自带，2023-01-28） |

版本不匹配导致无法插桩。已用最小 CUDA 程序（`cudaMalloc` + 一个 kernel）
复现相同的两类错误，**确认为环境问题，与本项目代码无关**。

## 对比：第 1 轮的 sanitizer 是真实运行的

`records/03-mxfp8-fused-scale/validation/memcheck.log`：

```
========= COMPUTE-SANITIZER
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

该记录产生于 2026-09-10，当时环境可用。现在环境已变化。

## 本轮替代证据

1. `--self-test`：`pipeline_cases=299 PASS`，三种输出类型 `element_mismatches=0`、
   `max_abs_diff=0`，与冻结 CPU oracle 逐字节比对。
2. `group_boundary.txt`：针对融合 kernel `width=8` 分组边界的专项测试 15/15 通过。
   构造组基准逐组放大、组内递增的输入，覆盖 16..8199 共 15 种规模（含非整组尾部）。
   若 shuffle width 或组首判断错误导致 scale 跨组串用，该测试必然失败。
3. `group_boundary_test.py`：测试脚本源码，可复现。
