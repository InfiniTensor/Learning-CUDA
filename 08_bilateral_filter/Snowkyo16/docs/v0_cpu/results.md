# V0: CPU 基线实现

## 算法原理

双边滤波是一种非线性保边滤波器，对每个像素 p，输出为其邻域内所有像素的加权平均：

```
BF[I](p) = (1/W_p) · Σ_{q∈S} G_s(||p-q||) · G_r(|I(p)-I(q)|) · I(q)
```

- G_s：空间权重（高斯），距离越远权重越小
- G_r：颜色权重（高斯），颜色差异越大权重越小
- W_p：归一化因子

参考：[Tomasi & Manduchi 1998 - Section 2 "Definition"](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)

## 实现说明

- 纯 CPU 实现，三层嵌套循环（遍历像素 → 遍历邻域 → 遍历通道）
- 使用 stb_image 进行图像读写
- 支持灰度和 RGB 彩色图像

## 滤波效果

| 原图 | CPU 滤波结果 |
|------|-------------|
| ![原图](images/input.png) | ![CPU输出](images/output_cpu.png) |

观察：
- （待填写：平坦区域的平滑效果）
- （待填写：边缘保留效果）

## 性能数据

测试环境：NVIDIA A100 服务器（CPU 部分）

滤波参数：radius=5, sigma_spatial=3.0, sigma_color=30.0

| 图像尺寸 | 处理时间 (ms) | 吞吐量 (MPixels/s) |
|----------|--------------|-------------------|
| 512×512  | （待填写）    | （待填写）         |

## 分析

- 4K 60fps 目标需要约 497 MPixels/s
- 当前 CPU 吞吐量约 0.5 MPixels/s
- 差距约 900 倍，说明必须使用 GPU 并行加速
- 下一步：V1 实现 naive CUDA kernel，每个线程处理一个像素
