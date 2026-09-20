# 硬件/库对照路径基准结果

本目录保存「硬件/库对照路径」（训练营方向二加分项）的实测数据。所有 JSON 都由
真实运行生成，可用下述命令复现。

## 环境

- GPU：NVIDIA GeForce RTX 5060 Ti（sm_120，Blackwell），驱动 596.49。
- 工具链：CUDA 12.9（`~/cuda-root/usr/local/cuda-12.9`）；cuBLAS/cuBLASLt 12.9
  （pip nvidia-cublas 包，`~/.local/lib/python3.12/site-packages/nvidia/cublas`）。
- 构建：`make BUILD_DIR=build-hw WITH_HW_PATH=1 CUDA_HOME=... CUDA_ARCH=120 -j`。
- 注意：采集时机器上有一个常驻 llama-server 占用约 13GB 显存（GPU 利用率
  接近 0 但存在偶发干扰），绝对带宽/算力数字可能略低于空载环境。

## 复现

```bash
./scripts/run_hw_benchmarks.sh    # 默认 1024/4096 两种尺寸，warmup 20 / iters 200
```

## 目录内容

- `cvt_<side>/hw_path.json`：MXFP8 元素级编解码，SM 原生 `cvt` 指令 vs 软件位
  操作 kernel 的时间、有效带宽和加速比；`encode_byte_mismatches` 与
  `decode_bit_mismatches` 为逐字节/逐位等价性比对结果（应为 0）。
- `gemm_mxfp8_<side>/hw_gemm.json`：推理 GEMM（Y = X·Wᵀ，M=N=K=side）对照。
  `hw_*` 为 cuBLASLt 原生 FP8 E4M3 + VEC32_UE8M0 块缩放 GEMM；
  `sw_dequant_ms` + `sw_gemm_ms` 为软件「反量化到 FP16 + cublasGemmEx」流水线。
- `gemm_nvfp4_<side>/hw_gemm.json`：同上，FP4 E2M1 + VEC16_UE4M3 块缩放，
  FP32 全局 scale 折进 alpha。
- 每个目录同时保留软件路径的 `quantized.lpq` / `dequantized.bin` /
  `metrics.json`，作为同输入下的软件基线。
- GEMM 数值校验：`self_check_*_rel_error` 是 256³ 上对 CPU double 参考的相对
  误差；`bench_hw_vs_sw_max_*` 是基准尺寸上两条路径输出的最大偏差。

GEMM 对照的量化本身由软件路径完成（两条路径共用同一份量化结果），比较的是
量化之后的推理计算段；scale 的 128×4 tile 重排属于一次性 setup，不计入计时。
