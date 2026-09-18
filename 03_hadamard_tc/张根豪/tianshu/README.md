# FastHadamard-CoreX

这是 `FastHadamard-CUDA` 面向天数智芯 CoreX CUDA 兼容环境的独立适配子目录。它复用上级目录中已经验证过的 Hadamard kernel 和 benchmark，不修改 CUDA 主工程；编译时通过 `COREX_HOME` 注入 CoreX 的 CUDA 兼容头和运行库。

## 环境

需要安装天数智芯 CoreX SDK。当前 CoreX 4.4 使用定制 Clang 的 `ivcore` 编译模式：

```bash
cd tianshu
source env.sh
make COREX_CXX=/path/to/corex/clang++
```

默认值适配本仓库验证环境：`COREX_HOME=/usr/local/corex-4.4.0`，`COREX_CXX=/usr/local/corex/bin/clang++`。`/usr/local/corex/bin/nvcc` 在该环境中只是版本探测脚本，不用于编译。

可通过 `COREX_EXTRA_FLAGS` 传入厂商 SDK 版本所需的额外参数：

```bash
make COREX_EXTRA_FLAGS="<CoreX 编译参数>"
```

## 构建与验证

```bash
make
make smoke
make clean
```

`smoke` 会运行小规模 FP16 Hadamard 变换并启用 CPU 参考结果校验。运行时使用 `COREX_HOME/lib64`，也可以通过 `COREX_LD_LIBRARY_PATH` 覆盖库路径。

当前适配会对 CoreX 不支持的 CUDA `__half2` 地址空间转换走标量低精度 load/store，仍保持 FP32 蝶形累加和原有低精度回写规则。

## 已验证环境

本目录已在 `Iluvatar MR-V100` 上完成验证：

- `make` 构建 `hadamard_bench` 和 `hadamard_advanced_bench` 成功；
- FP16/BF16、`head_dim=32/64/128/256/512/1024`、归一化开关共 24 组 CPU 参考核对全部 `PASS`；
- 高级路径在 FP16/BF16、`head_dim=64/128/256/512/1024` 上全部通过 `optimized_vs_baseline=BIT-EXACT`、`fused_vs_unfused=BIT-EXACT` 和 `gpu_vs_cpu_quant=BIT-EXACT`。

## 适配边界

CoreX 提供 CUDA 风格的 runtime、FP16/BF16 头和 warp intrinsic 时，上级 `src/` 可以直接复用。若具体芯片 SDK 不支持某个 CUDA intrinsic，应在本目录增加兼容头并通过 `COREX_EXTRA_FLAGS=-include...` 注入，不要修改 CUDA 主线源码。
