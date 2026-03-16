# NF4 Dequantization - Multi-Platform Support (NVIDIA & 国产芯片)

具体报告于nf4_report中, 这是一个实现了 QLoRA 4-bit NormalFloat (NF4) 动态反量化算子的项目。
当前工程不仅支持原生 NVIDIA GPU，还成功适配了国内主流的三大算力平台：
- **NVIDIA (NVIDIA GPU)**
- **Iluvatar (天数智芯)**
- **Moore Threads (摩尔线程)**
- **MetaX (沐曦)**

---

## 1. 环境准备 (Prerequisites)

在进行编译和测试之前，需要在各自平台/容器中安装必要的 Python 依赖以生成测试用例。测试数据生成脚本依赖于 `torch`、`numpy` （和可选的 `bitsandbytes`）。

```bash
# 推荐使用国内镜像源下载依赖 (必须确保 numpy 版本为 1.x 代以防止 PyTorch 不兼容)
pip3 install "numpy<2.0.0" torch bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall
```

## 2. 生成测试数据

在正式编译与运行算子之前，首先需要利用 PyTorch 和 Bitsandbytes 在本地生成模拟的 `test_weights.bin` 和真实基准参考文件 `ground_truth.bin` 以及配置 `params.txt` ：

```bash
python3 generate_test_data.py
```
> **注意**：如果在只搭载国产芯片且无正常 CUDA 执行库的镜像上，此脚本也可以无缝生成二进制文件用于后续的 C++ 端纯前向推理测试。

## 3. 多平台编译与测试指令

项目采用了一套统一的 `Makefile` 并通过 `PLATFORM` 变量实现平台路由。只需在 `make` 时通过 `PLATFORM=` 指定目标芯片厂商环境。

### 3.1 NVIDIA (默认平台)
```bash
make clean
# 编译
make PLATFORM=nvidia build
# 运行
./nf4_dequantizer
```

### 3.2 Iluvatar (天数智芯)
天数智芯平台使用 `clang++` (基于 LLVM) 和 `corex` 构建库。使用前请确保你已经通过 K8s 进入了包含天数 SDK `corex` 的容器中。
```bash
make clean
# 编译
make PLATFORM=iluvatar build
# 运行
./nf4_dequantizer
```

### 3.3 Moore Threads (摩尔线程)
摩尔线程平台基于 MUSA 核心架构，使用 `mcc` 编译并将自动使用 `.mu` 为拓展名的特化源码。
```bash
make clean
# 编译
make PLATFORM=moore build
# 运行
./nf4_dequantizer
```

### 3.4 MetaX (沐曦)
沐曦平台基于 MACA 核心架构，使用 `mxcc` 编译并将自动使用 `.maca` 为拓展名的特化源码。
```bash
make clean
# 编译
make PLATFORM=metax build
# 运行
./nf4_dequantizer
```

## 4. 特性与修改点 (Changelog)

- 移除了裸写 `cudaMallocHost` 的硬编码，取而代之为宏包装，兼容各个平台的 Pinned Memory 分配（如 `mcMallocHost`）。
- 针对沐曦使用内置的 `maca_bfloat16.h` 进行完整支持。
- 针对于摩尔线程 `__halves2musa_bfloat162` 缺失情况，使用了寄存器级位运算拼接（`bitwise packing`）完成平替保护。
