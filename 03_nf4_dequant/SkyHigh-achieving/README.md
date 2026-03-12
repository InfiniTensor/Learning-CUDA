# SkyHigh-achieving

本项目为 SkyHigh-achieving 项目的技术总结报告，包含实现思路、优化历程与性能分析。

## 📁 项目结构

```tree
SkyHigh-achieving/
├── Final_Project_Report.md
├── README.md
├── benchmark_vs_bnb.py
├── dequant_kernel.cu
├── dequant_kernel.h
├── dequant_kernel.ptx
├── dequant_kernel_v2.cu
├── main.cpp
└── run_log_remote.md
```

- **Final_Project_Report.md** → 详细的技术总结报告，包含实现思路、优化历程与性能分析
- **README.md** → 项目提交说明与文件结构介绍（本文件）
- **benchmark_vs_bnb.py** → 工业级对比脚本，用于对标 bitsandbytes 库的性能与精度
- **dequant_kernel.cu** → 核心 NF4 解量化 Kernel 实现（v4 优化版），包含 Packed Store 与 Bounds 优化
- **dequant_kernel.h** → Kernel 函数头文件定义，提供 C++ 调用接口
- **dequant_kernel.ptx** → NVCC 编译生成的 PTX 汇编代码，用于指令级分析
- **dequant_kernel_v2.cu** → 早期版本的 Kernel 实现（v2），用于性能对比参考
- **main.cpp** → C++ 测试驱动程序，包含随机数据生成、MAE 精度验证与基础性能测试逻辑
- **run_log_remote.md** → A100 服务器上的完整运行日志与性能实测数据记录
