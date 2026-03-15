# 摩尔线程 (MUSA) 适配版 NF4 反量化

该目录是对 `kernel/` 中 CUDA 版本的平移适配，目标是运行在摩尔线程 GPU 环境。

## 适配要点

- 编译器从 `nvcc` 切换为 `mcc`
- 主源码使用 `.mu` 后缀
- 运行时 API 使用 `musa*`（例如 `musaMalloc` / `musaMemcpy`）
- 保留与原工程一致的二进制输入输出格式，直接复用 `scripts/verify.py`

## 目录文件

- `main.mu`: 主入口，负责文件 IO / kernel 启动 / 性能统计
- `nf4_dequant_kernel.mu`: NF4 反量化 kernel
- `Makefile`: 使用 `mcc` 构建 `nf4_dequant_musa`
- `run_moore.sh`: 一键 build/run/verify

## 构建

```bash
cd kernel_noncuda/moore
make MCC=mcc -j
```

## 运行

```bash
# 需要已有 data/nf4_weights_*.bin
./nf4_dequant_musa ../../data/nf4_weights_4096x4096_bs64.bin \
                   ../../data/moore_output_4096x4096_bs64_fp16.bin \
                   fp16 10 100
```

## 一键流程

```bash
cd kernel_noncuda/moore
bash run_moore.sh test --rows 4096 --cols 4096 --blocksize 64 --compute_type fp16
```

说明：

- `run_moore.sh` 默认只消费已有测试数据，不会调用 `generate_data.py`。
  - 可先在另一台 CUDA 机器执行 `./run.sh generate` 生成 `data/` 再拷贝过来。
- `compute_type` 支持 `fp16` 和 `bf16`，输出文件格式与原验证脚本兼容。
