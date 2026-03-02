# 沐曦 NF4 反量化

## 1) 编译

优先使用 `mxcc`（沐曦环境），没有则回退 `nvcc`（CUDA 兼容环境）。

```bash
cd 03_nf4_dequant/mx

# 方式1：脚本自动选编译器
bash run_all.sh

# 方式2：手动编译
mxcc -O3 -std=c++17 -o nf4_dequant_mx nf4_dequant_mx.cu
# 或
nvcc -O3 -std=c++17 -o nf4_dequant_mx nf4_dequant_mx.cu \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
```

## 2) 单文件运行

```bash
./nf4_dequant_mx ../weight_data/weight_1024x1024_bs64.bin
```

输出：
- 解量化结果：`../mx_results/dequant_<shape>_bs<block>.fp16`
- 性能日志：`../mx_results/perf_<shape>_bs<block>.log`

## 3) 批量运行

```bash
bash run_all.sh
```

## 4) 功能测试（CPU 参考对比）

```bash
python test_nf4_dequant_mx.py
```

可选参数：

```bash
python test_nf4_dequant_mx.py --rows 1024 --cols 768 --blocksize 64 --tol 0.02
```

## 5) 与 BnB 结果对比

先保证 `../bnb_results` 与 `../bnb_benchmark_results.csv` 已存在。

```bash
python compare_results.py
```

将生成：
- `comparison_mx_results.csv`
- `comparison_mx_results.md`
