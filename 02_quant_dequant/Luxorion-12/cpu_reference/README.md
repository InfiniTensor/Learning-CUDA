# cpu_reference

CPU参考实现用于先把格式规则、分块、scale、编码、打包、恢复和指标跑通，强调可读性与完整边界检查，不以性能为目标。

- `include/`：CPU接口以及CPU/GPU共同使用的结果数据结构。
- `src/`：CPU算法、固定20字节tensor文件IO与演示入口。
- `tests/`：独立oracle、文件协议、边界与错误输入测试。

此目录不包含CUDA头文件或`.cu`源文件，默认构建无需CUDA环境。

量化采用行主序连续展平分块：MXFP8每32个连续元素共享scale，NVFP4每16个连续元素共享局部scale，量化块允许跨越行边界。
