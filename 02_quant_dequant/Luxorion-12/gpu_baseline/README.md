# gpu_baseline

GPU正确性baseline包含MXFP8与NVFP4的CUDA量化/反量化。CUDA接口、kernel和CPU/GPU对照测试均位于本目录；CPU reference只作为结果数据契约和独立oracle被引用。

- `include/`：CUDA专用入口。
- `src/`：baseline kernel与当前host包装接口。
- `tests/`：逐层比较scale、编码、packed data和恢复FP32。
- `tools/generate_inputs.cpp`：生成并回读校验固定性能输入与manifest。
- `tools/benchmark_main.cpp`：执行固定协议，校验CPU/GPU并输出CSV和JSON。

当前版本优先保证可读性与正确性。`benchmark.h`提供device-resident性能接口：显存只分配一次，分别统计量化kernel、反量化kernel、量化+反量化显存内pipeline，以及H2D+pipeline+D2H端到端时间。kernel与pipeline使用CUDA event，端到端使用host `steady_clock`；文件IO、分配和CPU校验均不进入时间。尚未加入warp shuffle、向量化或快速编码器。

GPU与CPU reference采用相同的连续展平分块语义：MXFP8通过`i / 32`、NVFP4通过`i / 16`定位量化块，不在行边界重新开始分块。

性能输入默认生成到`input/benchmark/`，固定使用seed=1234。FP32与FP16均覆盖uniform、normal、outliers、90%零值的zero-heavy、块间量级变化的multiscale，以及元素总数不能被16/32整除的tail-normal。它们是可复现的合成覆盖集，不替代真实模型权重或激活。生成数据的时间不进入GPU计时。

输入文件只需生成一次；benchmark只读取已有文件，不会自动重建：

```powershell
.\build\cuda\Release\generate_gpu_baseline_inputs.exe
```

正式协议先预热10次，再重复100次，报告median与nearest-rank p95。运行：

```powershell
.\build\cuda\Release\gpu_baseline_benchmark.exe
```

结果写入`output/gpu_baseline/baseline_results.csv`和`baseline_results.json`。本次RTX 3050 Laptop GPU结果的量化kernel median：MXFP8为18.19–18.69 ms，NVFP4为25.78–26.30 ms；三类输入的GPU编码和恢复值均与CPU oracle完全一致。这些数值只用于本机baseline，跨设备比较时应重新运行相同协议。
