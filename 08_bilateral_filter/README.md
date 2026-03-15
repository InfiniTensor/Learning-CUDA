# 实时双边滤波-CUDA 

## 项目结构

当前项目结构如下：

```txt
实时双边滤波-CUDA/
├─ 朴素实现代码及结果/
├─ 实时双边滤波-CUDA论文.pdf
├─ 项目要求.md
├─ baboon.bmp
├─ girl.bmp
├─ main.cu
├─ monarch.bmp
├─ params.txt
└─ README.md
```

其中：

- `main.cu`：共享内存 + 网格跨步循环优化后代码
- `params.txt`：双边滤波参数文件
- `baboon.bmp`、`girl.bmp`、`monarch.bmp`：测试图像
- `实时双边滤波-CUDA论文.pdf`：最终提交的总结报告 PDF
- `项目要求.md`：课程项目要求
- `朴素实现代码及结果/`：朴素版本代码及阶段性结果记录

## 英伟达平台编译与运行

在已经安装 CUDA 和 OpenCV 的英伟达平台上，可以直接使用下面的命令编译并运行：

```bash
nvcc -std=c++11 main.cu -o bilateral_filter \
-I/usr/local/include/opencv4 \
-L/usr/local/lib \
-lopencv_core \
-lopencv_imgproc \
-lopencv_imgcodecs

./bilateral_filter <测试图像名称> params.txt
```

例如：

```bash
./bilateral_filter monarch.bmp params.txt
```

## 参数文件格式

`params.txt` 示例：

```txt
radius = 5
sigma_spatial = 3.0
sigma_color = 30.0
use_adaptive_radius = false
```

## 输出内容

程序运行后会输出：

- 滤波后的 raw 文件
- 滤波后的图像文件
- 性能日志
- 终端中的 GPU/CPU 时间、吞吐量、加速比和 MAE
