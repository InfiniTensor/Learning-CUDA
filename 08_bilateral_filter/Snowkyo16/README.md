# 基于CUDA实现实时图像双边滤波

## 编译与运行

```bash
# 编译 + 运行（自动选取 test_images/ 下第一张图片）
make

# 仅编译
make build

# 指定图片运行
make run INPUT=test_images/lena.png

# 清理
make clean
```

## OpenCV 对比验证

```bash
# 自动读取 params.txt 参数
python3 scripts/compare_opencv.py test_images/lena.png output/images/lena_cpu.png

# 手动指定参数文件
python3 scripts/compare_opencv.py test_images/lena.png output/images/lena_cpu.png params.txt
```

