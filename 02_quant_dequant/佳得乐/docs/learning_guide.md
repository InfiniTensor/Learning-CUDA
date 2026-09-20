# 学习指南

建议按以下顺序阅读和手写核心函数；文件容器、命令行与实验管理作为工程辅助复用，不必全部重写。

## 1. 标量编码

`include/numeric.hpp`：`decode` 输入编码与 format(0/1/2)，输出 double；`encode` 输入 double、舍入和逻辑索引，输出真实低位宽编码。
手算 E2M1 正数表 `0,0.5,1,1.5,2,3,4,6`，所以 1.25 在 1 与 1.5 的中点，RNE 选择编码 2，即1。随机舍入 1.125 时选1.5概率为1/4。
运行 `./build/scalar_tests`。练习手写 decode 和相邻值 RNE；再对照 `tests/scalar.cpp::oracle` 的独立整数权重求和参考，而不是拿自己的解码验证自己。

FP16/BF16 输出由 `half_bits`/`bf_bits` 完成，无 BF16 原生运算依赖。它们是工程辅助，但建议理解保留位最低位决定 ties-to-even 的原因。

## 2. scale

`mxscale` 输入块 amax，输出 E8M0 字节；E4M3 的 amax=448 得 scale字节127（乘数1），略大于448得128（乘数2）。
`global_scale`、`nvscale`：若全局最大值2688，则 g=1；局部最大值6 得 s=1，即E4M3编码56。量化6得到E2M1编码7，解码7×1×1=6。
重要练习：先编码局部 scale，再用它的解码值除输入。不要分别使用“原始 scale”和“存储 scale”。
测试文件 `tests/scalar.cpp` 和 `tests/integration.cpp` 覆盖全零、常数、混合符号、离群值、极值及非法 scale。

## 3. 打包与解包

`src/cpu.cpp::quant_cpu` 的字节循环中，两个 E2M1 值共用一个 uint8。输入编码 `[2,3,7]` 应写成 `[0x32,0x07]`，最后高半字节0。没有行尾 padding。
GPU `encode_kernel` 和 `decode_kernel` 每个线程负责一个完整字节，避免两个线程 read-modify-write 同一地址。
建议手写打包/解包的十行循环，运行 `./build/integration_tests` 验证奇数长度和3×17形状。

## 4. CPU 完整流程

`src/cpu.cpp::quant_cpu(Tensor,Config)->Packed`：检查有限输入→全局 amax→块 scale→编码/打包。 块沿行方向划分：每行独立按 block_size 切块，块号 b = r * ceil(cols/block_size) + c/block_size。
`dequant_cpu(Packed)->Tensor`：读取 scale 与低精度码→乘法恢复→输出类型转换。
`src/io.cpp` 提供配置解析和稳定 little endian 容器；理解字节偏移和长度验证即可，无需手写所有 CLI。

```bash
mkdir -p results/manual
./build/lp generate --rows 3 --cols 17 --dtype fp16 --distribution normal --seed 42 --output results/manual/input.tensor
./build/lp quantize --input results/manual/input.tensor --config configs/e5m2.toml --backend cpu --output results/manual/weights.lp
./build/lp dequantize --input results/manual/weights.lp --backend cpu --output results/manual/restored.tensor
./build/lp verify --input results/manual/input.tensor --packed results/manual/weights.lp --backend cpu
```

## 5. CUDA baseline

`src/gpu.cu`：`reduce_all` 256线程共享内存树归约，网格步长分散大张量工作；块结果对非负 float 位模式做 atomicMax。`make_global` 只做常数工作，不扫描张量。
`make_scales<false>` 每个warp处理一个量化块，lane负责元素，shuffle取最大值；NV只有16个有效lane，其余补零。块号到元素的映射按行方向反推：先由 b 算出所在行 r 和行内块号 cb，再取该块覆盖的列范围。tensor模式读取已经并行求得的全局最大值。
`encode_kernel<false>` 每线程处理一个 packed byte；`decode_kernel<false>` 同样从一个字节恢复一个/两个元素。元素 i 的块号由 (i/cols)*bpr + (i%cols)/block 得到。尾部必须先检查逻辑元素边界再读写。
重点手写归约和字节线程映射，复用 `Device`/`Events` 的资源管理。
运行 `./build/integration_tests gpu` 会同时核对 baseline 与 optimized 的 scale、packed、输出，而不仅比较误差容差。


## 6. 优化版本

`decode_fast` 利用三种元素全部精确可表示为 FP32 的事实，直接构造 FP32 位模式；`encode_fast` 保留同样的二分、距离和随机数，避免内循环反复做 `ldexp`。
`make_scales<true>` 也使用快速 E4M3 编码。优化没有改变块大小、随机索引或 scale 规则。
建议先理解性能证据，再尝试把 scale+encode 融合；不要为了 kernel 数量少而破坏 tensor scaling 的全局依赖。

## 7. 性能分析

`src/main.cpp` benchmark：CPU steady_clock；GPU同轮 CUDA Events 的边界划分 H2D、scale、encode、decode、D2H；外层墙钟计完整API；文件往返另计。
`tests/quality.cpp` 输出全配置重建误差；`scripts/reproduce.py` 保存命令与环境，`scripts/summarize.py` 从真实 CSV 计算均值/标准差与图表。
运行 `python3 scripts/reproduce.py quick`。阅读 `results/<run_id>/summary.md` 和 `metadata.json`，解释为什么小矩阵中 GPU API、分配与 I/O 开销可能超过计算收益。带宽是逻辑最小字节流量，不等于 DRAM counters；ncu无权限不能凭空补数。
