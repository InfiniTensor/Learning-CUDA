# MUSA 诊断工具

工具与正式算子分开，以下命令在项目根目录执行。计时和 trace 不替代 CPU 数值验证。

## 不依赖硬件计数器的分阶段计时

```bash
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_22 -std=c++17 -O3 -ffp-contract=off \
  -DLP_BACKEND_MUSA -DTEST_NVFP4 -ICore/backends/musa/include -ICore \
  Core/backends/musa/tools/stage_profile.cu -L/usr/local/musa/lib -lmusart \
  -Wl,-rpath,/usr/local/musa/lib -o Core/build-musa/stage_nvfp4
Core/build-musa/stage_nvfp4
```

移除 `-DTEST_NVFP4` 并改输出名称即可测 MXFP8。固定 4M 输入、预热 3 次、测量 20 次；依次输出 maximum、finalize_max、融合量化、FP32 反量化的 median/P95。每阶段各自插入 event，因此耗时之和不能当成完整设备序列的精确时间；正式性能仍以 benchmark 为准。

## MUPTI 活动采集（需要宿主机开放 MT-Perf）

```bash
g++ -std=c++17 -shared -fPIC Core/backends/musa/tools/activity_trace.cpp \
  -I/usr/local/musa/include -L/usr/local/musa/lib -lmupti -pthread \
  -Wl,-rpath,/usr/local/musa/lib -o Core/build-musa/liblp_mupti.so
LP_MUPTI_TRACE=records/new-trace.tsv LD_PRELOAD=$PWD/Core/build-musa/liblp_mupti.so \
  Core/build-musa/pipeline_mxfp8 --benchmark 4194304 5
```

输出 TSV 包括 kernel 时间戳、grid/block、寄存器和共享/局部内存字段。末尾会打印记录数和丢失数，只有记录非零、丢失数为零且时间戳有效才能分析。不存在的输出文件才可创建，避免覆盖旧证据。

当前远程容器在 2026-09-15 实测报 `MT-Perf not able to establish hw connection`，随后进程异常退出，记录数为 0。因此插件目前仅完成编译和失败诊断，尚未验证成功采集，不能拿空报告判断计算或访存受限。需平台管理员核对 MT-Perf 服务/驱动支持及容器设备、权限映射后重试。

## 内存检查状态

当前安装目录未找到 MUSA 专用设备 memcheck/racecheck CLI；`mcc --help-hidden` 中的 `-fgpu-sanitize` 标明仅用于 AMDGPU，不能据此声称支持 MUSA 设备 ASan。需平台方提供匹配驱动/SDK 的检查工具。现有 CTest 覆盖尾部输出哨兵、CPU 对照和同步边界，但不等价于完整越界读/数据竞争检测。
