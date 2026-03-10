# 基于CUDA实现实时图像双边滤波

## 版本说明

| 版本 | 说明 | MODE 参数 |
|------|------|-----------|
| v0_cpu | CPU 基线实现 | `v0` |
| v1_naive | GPU Naive，一个线程一个像素 | `v1` |
| v2_smem | GPU Shared Memory Tiling | `v2` |
| v3_constmem | GPU Constant Memory LUT | `v3` |
| all | 跑所有版本 + 性能对比表 | `all`（默认） |


## 编译与运行

```bash
# 清理
make clean

# 仅编译
make build

# 运行（默认跑全部版本)
make run

# 运行（可以指定空闲 GPU）
CUDA_VISIBLE_DEVICES=2 make run

# 只跑 CPU 版本（不需要 GPU）
make run MODE=v0

# 只跑 GPU V1/V2/V3 版本
make run MODE=v1
make run MODE=v2
make run MODE=v3

# 指定图片
make run INPUT=test_images/yosemite.jpg MODE=all

```


## OpenCV 对比验证

**通过标准:** MAE < 1，PSNR > 40 dB

```bash
# 验证各版本输出
python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v0_cpu.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v1_naive.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v2_smem.png

python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v3_constmem.png
```


## 性能分析 (nsys)

```bash
# 采集不同版本的 nsys profile
# 以 V3 为例（指定空闲 GPU）
CUDA_VISIBLE_DEVICES=2 nsys profile --trace=cuda -o output/v3_yosemite make run INPUT=test_images/yosemite.jpg MODE=v3

# 查看统计报告
nsys stats output/v3_yosemite.nsys-rep
```
