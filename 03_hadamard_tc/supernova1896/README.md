# Hadamard 变换加速

这是 2026 夏季训练营 CUDA 方向选题三的独立实现项目。项目实现了面向
`[batch_size, seq_len, num_heads, head_dim]` 张量的归一化快速 Hadamard
变换（Fast Hadamard Transform, FHT），其中 `head_dim` 需要是 2 的幂，当前
支持并测试了 `64`、`128`、`256`。

## 功能

- FP16 输入/输出路径：使用 shared memory 实现 FHT。
- BF16 输入/输出路径：使用 `uint16_t` 保存 BF16 位模式，并在软件中完成转换，
  因此基础版本不依赖原生 BF16 Tensor Core。
- FP16 WMMA/Tensor Core 对照路径：计算 `X * H / sqrt(head_dim)`。
- Hadamard + signed INT4 融合量化：输出为真实 4 bit 打包格式，每个字节存两个元素。
- Hadamard + 软件 E4M3 FP8 编码融合路径。
- CPU FHT 参考实现，用于正确性测试。
- CPU FHT 性能基线，用于计算 GPU 加速比。
- 融合量化一致性测试：验证 fused kernel 与“先 Hadamard、再量化”的结果一致。
- benchmark 命令行程序，输出 CSV 性能日志。
- Nsight Systems / Nsight Compute 性能分析脚本和说明。

## 构建

使用 CMake：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

如果需要显式指定 GPU 架构：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release
```

如果环境中没有 CMake，但可以直接使用 `nvcc`，Linux 下也可以这样编译核心程序：

```bash
nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude src/hadamard_cuda.cu tests/test_hadamard.cu -o hadamard_tests
nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude src/hadamard_cuda.cu src/hadamard_bench.cu -o hadamard_bench
g++ -std=c++17 -O3 src/hadamard_cpu_bench.cpp -o hadamard_cpu_bench
```

其中 `sm_89` 可按实际 GPU 架构替换，例如 Turing T4 可使用 `sm_75`。

## 测试

CMake 构建后的 Windows 路径：

```powershell
.\build\Release\hadamard_tests.exe
```

直接 `nvcc` 构建后的 Linux 路径：

```bash
./hadamard_tests
```

测试覆盖 `head_dim = 64, 128, 256` 下的 FP16 和 BF16：

- FP16 最大绝对误差阈值：`1e-2`
- BF16 最大绝对误差阈值：`5e-2`
- fused INT4：packed bytes 和 scale 与“先变换后量化”完全一致
- fused E4M3：编码字节与“先变换后量化”完全一致

## 性能测试

```powershell
.\build\Release\hadamard_bench.exe --batch 8 --seq 1024 --heads 16 --dim 128 --dtype fp16 --repeats 100 --log perf.csv
.\build\Release\hadamard_bench.exe --batch 8 --seq 1024 --heads 16 --dim 128 --dtype bf16 --repeats 100 --log perf.csv
```

CPU 基线：

```bash
./hadamard_cpu_bench --rows 16384 --dim 128 --repeats 10
```

CSV 日志包含：

- kernel 名称
- dtype 和输入形状
- 平均 kernel 时间，单位 ms
- 有效内存带宽，单位 GB/s

## 实现说明

主实现是 shared-memory FHT：每个 CUDA block 处理一个展平后的
`(batch, seq, head)` 行向量，在 shared memory 中执行蝶形加减，最后乘以
`1 / sqrt(head_dim)` 完成正交归一化。这个方案避免显式构造 Hadamard 矩阵，
算术复杂度为 `O(D log D)`。

WMMA 路径作为 FP16 Tensor Core 对照实现，会显式构造缩放后的 Hadamard 矩阵，
再用分块矩阵乘法计算。它展示了 Tensor Core 用法，但复杂度为
`O(D^2)`，对于本题中一行一个 head vector 的工作负载，shared-memory FHT 通常
更贴合算法结构。

## 性能分析

Linux 环境可以使用以下脚本重新编译、测试并采集 profiling：

```bash
chmod +x scripts/profile_linux.sh
CUDA_ARCH=89 ./scripts/profile_linux.sh
```

详细结果见 [PROFILING.md](PROFILING.md)。其中：

- `cuobjdump` 已确认 WMMA kernel 生成了 `HMMA.16816.F32` Tensor Core 指令。
- Nsight Systems 可以正常生成 CUDA kernel 时间线。
- Nsight Compute 在当前远端容器中受到 `ERR_NVGPUCTRPERM` 权限限制，脚本会保留
  失败日志，不会伪造硬件计数器结果。

## 沐曦曦云 C500 适配

项目提供独立的 `.maca` 源文件，不要求用 `mxcc` 直接把 NVIDIA `.cu` 文件当作
C500 主实现：

- `src/hadamard_maca.maca`：C500 的 FP16/BF16 FHT、融合 INT4 和融合 E4M3 kernel。
- `src/hadamard_bench_maca.maca`：C500 benchmark 入口，不调用 NVIDIA WMMA。
- `tests/test_hadamard_maca.maca`：C500 正确性入口。
- `scripts/build_maca.sh`：使用 `mxcc` 的一键构建脚本。

C500 版本使用 CUDA 兼容运行时接口，但链接时必须同时加入
`libruntime_cu.so` 和 `libsymbol_cu.so`，否则会出现 `wcuda*` 未定义符号。
脚本已经自动处理该链接要求。

在 C500 主机上构建和测试：

```bash
export MACA_PATH=/opt/maca
chmod +x scripts/build_maca.sh
./scripts/build_maca.sh
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
./build-maca/hadamard_tests_maca
./build-maca/hadamard_bench_maca \
  --batch 2 --seq 512 --heads 16 --dim 128 \
  --dtype fp16 --repeats 50 --log maca_perf_fp16.csv
./build-maca/hadamard_bench_maca \
  --batch 2 --seq 512 --heads 16 --dim 128 \
  --dtype bf16 --repeats 50 --log maca_perf_bf16.csv
```

也可以使用 CMake：

```bash
MACA_PATH=/opt/maca cmake -S . -B build-cmake-maca \
  -DHADAMARD_ENABLE_MACA=ON -DMACA_PATH=/opt/maca
cmake --build build-cmake-maca -j2
```

如果需要一次性生成 C500 正确性和性能日志：

```bash
chmod +x scripts/profile_maca.sh
./scripts/profile_maca.sh
```

C500 适配不调用 NVIDIA WMMA；WMMA/Tensor Core 证据只属于 NVIDIA 的独立对照
路径。C500 主路径采用共享内存蝶形 FHT，避免依赖 NVIDIA 专有头文件。

## 摩尔线程 MTT S4000 适配

项目提供独立的 MUSA `.mu` 源文件：

- `src/hadamard_musa.mu`：S4000 的 FP16/BF16 FHT、融合 INT4 和融合 E4M3 kernel。
- `src/hadamard_bench_musa.mu`：MUSA benchmark 入口。
- `tests/test_hadamard_musa.mu`：MUSA 正确性入口。
- `include/hadamard_musa.muh`：MUSA 运行时接口和公共声明。
- `scripts/build_musa.sh`：使用 `mcc` 的一键构建脚本。

MUSA 版本使用位级 FP16/BF16 转换，不依赖 NVIDIA `half`、`cuda_runtime.h`
或 WMMA。MUSA 4.3 的典型构建方式如下：

```bash
export MUSA_HOME=/usr/local/musa-4.3.6
export MUSA_ARCH_LIST=22
chmod +x scripts/build_musa.sh
./scripts/build_musa.sh
export LD_LIBRARY_PATH=${MUSA_HOME}/lib:$LD_LIBRARY_PATH
./build-musa/hadamard_tests_musa
./build-musa/hadamard_bench_musa \
  --batch 2 --seq 512 --heads 16 --dim 128 \
  --dtype fp16 --repeats 50 --log musa_perf_fp16.csv
./build-musa/hadamard_bench_musa \
  --batch 2 --seq 512 --heads 16 --dim 128 \
  --dtype bf16 --repeats 50 --log musa_perf_bf16.csv
```

也可以使用 CMake：

```bash
MUSA_HOME=/usr/local/musa-4.3.6 cmake -S . -B build-cmake-musa \
  -DHADAMARD_ENABLE_MUSA=ON -DMUSA_HOME=/usr/local/musa-4.3.6
cmake --build build-cmake-musa -j2
ctest --test-dir build-cmake-musa --output-on-failure
```

MTT S4000 适配不宣称使用 NVIDIA Tensor Core；主路径采用 MUSA shared-memory
蝶形 FHT，保证国产 GPU 运行时和 NVIDIA CUDA 对照路径相互独立。
