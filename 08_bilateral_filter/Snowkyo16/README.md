# 基于CUDA实现实时图像双边滤波

## 编译与运行

```bash
# 仅编译
make build

# 运行（默认跑全部版本，需指定空闲 GPU）
CUDA_VISIBLE_DEVICES=2 make run

# 只跑 CPU 版本（不需要 GPU）
make run MODE=v0

# 只跑 GPU V1
CUDA_VISIBLE_DEVICES=2 make run MODE=v1

# 指定图片
CUDA_VISIBLE_DEVICES=2 make run INPUT=test_images/lena.png MODE=all

# 清理
make clean
```

## 版本说明

| 版本 | 说明 | MODE 参数 |
|------|------|-----------|
| V0 | CPU 基线实现 | `v0` |
| V1 | GPU Naive，一个线程一个像素 | `v1` |
| 全部 | 跑所有版本 + 性能对比表 | `all`（默认） |

## OpenCV 对比验证

```bash
# 验证 V0 CPU 输出
python3 scripts/compare_opencv.py test_images/lena.png output/images/lena_v0_cpu.png

# 验证 V1 GPU 输出
python3 scripts/compare_opencv.py test_images/lena.png output/images/lena_v1_naive.png
```

通过标准：MAE < 1，PSNR > 40 dB
