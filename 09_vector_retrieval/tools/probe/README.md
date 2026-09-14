# 设备能力探测（device_probe）

`device_probe.cu` 是报告 §3.3 与 §7.2 中“平台差异”结论的最小复现用例，检查两项
直接影响向量检索正确性的设备能力：

1. `atomicAdd` 是否真正生效（32 位与 64 位）——倒排表的计数与写入游标依赖它；
2. device 端 `double` / `float` 长求和精度——距离累计的累加类型依赖它。

## 编译与运行

```bash
# NVIDIA
nvcc device_probe.cu -o probe -lcublas && ./probe

# 天数智芯 CoreX
/usr/local/corex/bin/clang++ -x ivcore --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include device_probe.cu \
    -L/usr/local/corex/lib64 -lcudart -lcublas -o probe
LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib ./probe

# 沐曦 MetaX
/opt/maca/tools/cu-bridge/bin/cucc device_probe.cu \
    -I/opt/maca/tools/cu-bridge/include -I/opt/maca/include \
    -L/opt/maca/lib -lmcblas -o probe
LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/lib64 ./probe

# 摩尔线程 MUSA（注意：探测程序用的是 CUDA 名称，MUSA 下需把 cuda*/cublas*
# 替换为 musa*/mublas*，或直接复用 engine.cu 中的名称映射）
/usr/local/musa/bin/mcc -x musa --musa-path=/usr/local/musa device_probe_musa.mu \
    -I/usr/local/musa/include -L/usr/local/musa/lib -lmusart -lmublas -o probe
LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64 ./probe
```

## 本实验观察到的结果

| 探测项 | NVIDIA / 沐曦 / 摩尔线程 | 天数智芯 CoreX |
| --- | --- | --- |
| `atomicAdd(unsigned int*)` | OK | OK |
| `atomicAdd(unsigned long long*)` | OK（读到 256） | 调用不报错，但读到 0 |
| device `double` 长求和（128 维） | 与 CPU 参考一致 | 偏差约 1e-4 相对量级 |
| device `float` 长求和 | 与 CPU 参考差约 1e-5 | 与 CPU 参考一致 |

据此，程序在 CoreX 构建下把倒排表计数/游标改为 32 位原子操作、把距离累计改为
float；NVIDIA、沐曦与摩尔线程的构建保持 64 位原子与 double 累计。详见报告 §7.2。
