# 天数智芯 (Iluvatar) 适配版 NF4 反量化

该目录是对 `kernel/` 中 CUDA 版本的平移适配，目标是在天数智芯环境优先使用 CUDA 兼容编译链跑通。

当前策略：

- 默认编译器使用 `clang++`（可通过 `ILCC` 覆盖）
- 源码保持 `.cu` 形式，便于复用 CUDA 风格 kernel 与运行时 API
- 运行时 API 维持 `cuda*` 命名，依赖目标机提供 CUDA 兼容 SDK
- 保留与原工程一致的二进制输入输出格式，可直接复用 `scripts/verify.py`

## 目录文件

- `main.cu`: 主入口，负责文件 IO / kernel 启动 / 性能统计
- `nf4_dequant_kernel.cuh`: 适配后的 NF4 反量化 kernel
- `Makefile`: 使用 `ILCC` 构建 `nf4_dequant_iluvatar`
- `run_iluvatar.sh`: 一键 build/run/verify

## 构建

```bash
cd kernel_noncuda/iluvatar
make ILCC=clang++ -j
```

## 运行

```bash
# 需要已有 data/nf4_weights_*.bin
./nf4_dequant_iluvatar ../../data/nf4_weights_4096x4096_bs64.bin \
                       ../../data/iluvatar_output_4096x4096_bs64_fp16.bin \
                       fp16 10 100
```

## 一键流程

```bash
cd kernel_noncuda/iluvatar
bash run_iluvatar.sh test --rows 4096 --cols 4096 --blocksize 64 --compute_type fp16
```

说明：

- `run_iluvatar.sh` 默认只消费已有测试数据，不会调用 `generate_data.py`。
  - 可先在 CUDA 机器执行 `./run.sh generate` 生成 `data/` 再拷贝到目标机。
- `compute_type` 支持 `fp16` 和 `bf16`，输出文件格式与原验证脚本兼容。