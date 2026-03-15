# 沐曦 (MACA) 适配版 NF4 反量化

该目录是对 `kernel/` 中 CUDA 版本的平移适配，目标是运行在沐曦 GPU 环境。

## 适配要点

- 编译器从 `nvcc` 切换为 `mxcc`
- 主源码使用 `.maca` 后缀
- 运行时 API 使用 `mc*`（例如 `mcMalloc` / `mcMemcpy`）
- 保留与原工程一致的二进制输入输出格式，直接复用 `scripts/verify.py`

## 目录文件

- `main.maca`: 主入口，负责文件 IO / kernel 启动 / 性能统计
- `nf4_dequant_kernel.maca`: NF4 反量化 kernel
- `Makefile`: 使用 `mxcc` 构建 `nf4_dequant_maca`
- `run_mutex.sh`: 一键 build/run/verify

## 构建

```bash
cd kernel_noncuda/mutex
make MXCC=mxcc -j
```

## 运行

```bash
# 需要已有 data/nf4_weights_*.bin
./nf4_dequant_maca ../../data/nf4_weights_4096x4096_bs64.bin \
                   ../../data/mutex_output_4096x4096_bs64_fp16.bin \
                   fp16 10 100
```

## 一键流程

```bash
cd kernel_noncuda/mutex
bash run_mutex.sh test --rows 4096 --cols 4096 --blocksize 64 --compute_type fp16
```

说明：

- `run_mutex.sh` 默认只消费已有测试数据，不会调用 `generate_data.py`。
  - 可先在另一台 CUDA 机器执行 `./run.sh generate` 生成 `data/` 再拷贝过来。
- `compute_type` 支持 `fp16` 和 `bf16`，输出文件格式与原验证脚本兼容。
