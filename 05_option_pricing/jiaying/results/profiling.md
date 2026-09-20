# 本机性能分析与内存检查记录

执行日期：2026-09-19。所有命令从项目目录运行。这里区分成功采集、缺失数据和工具初始化失败。

## Nsight Systems：CUDA API 采集成功，GPU activity 缺失

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o build/pricing_profile \
  ./build/option_pricing --option examples/asian_call.cfg \
  --simulation examples/simulation.cfg --backend gpu --output build/profile_price

# 发行版 nsys 未自动找到 importer；显式调用现有同版本转换器。
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  -i build/pricing_profile.qdstrm -o build/pricing_profile.nsys-rep -f

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --force-export=true build/pricing_profile.nsys-rep
```

- `nsys_capture.txt`：原始采集输出，初次自动转换失败。
- `nsys_import.txt`：显式转换成功记录。
- `nsys_stats.txt`：CUDA API 汇总成功；GPU kernel 和内存活动表为空。
- 两次 launch 对应模拟/块归约融合 kernel 和最后一级归约 kernel。
- 一次 D2H memcpy 只复制 96 字节的统计量，路径不落全局数组。
- `cudaLaunchKernel` 主机耗时 13.108 ms；第一次 launch 13.033 ms，第二次 0.075 ms。可以确认初次提交成本显著，但没有足够 GPU 活动数据进一步定位设备执行瓶颈。
- profiler 运行与正式基准分开，其耗时不混入 `benchmark.csv`。

二进制报告保留在本地 `build/`，提交包只包含可读日志。未声称取得 GPU occupancy 或硬件计数器。

## Compute Sanitizer：未能初始化

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/option_pricing --option examples/asian_call.cfg \
  --simulation build/validation/layout_96_sim.cfg --backend gpu \
  --output build/sanitized_asian
```

`results/memcheck.txt` 记录工具要求开启 WDDM debugger interface，并报告 device not supported。程序正常完成不代表内存检查完成。没有修改 Windows 注册表、驱动配置或调试权限。最终结论：**该环境下 memcheck 未验证**。

## Nsight Compute：安装缺失规则目录

```bash
/usr/lib/x86_64-linux-gnu/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu \
  --set basic --launch-count 2 ./build/option_pricing \
  --option examples/asian_call.cfg --simulation examples/simulation.cfg \
  --backend gpu --output build/ncu_price
```

`results/ncu.txt` 记录默认 sections/rules 路径不存在，工具未开始有效采集。没有将该失败报告解释为应用错误或性能计数器结论。
