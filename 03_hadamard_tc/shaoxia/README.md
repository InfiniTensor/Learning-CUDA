# hadamard-bench

完整的实现思路、优化过程、精度验证与性能分析见 [项目报告](docs/report.md)。

面向 NVIDIA GPU 的纯 C++/CUDA Hadamard 变换实验，包括：

- CUDA Core Hadamard：scalar、8-wide vector、multi-warp 和自动 dispatch；
- 与 vendored `fast-hadamard-transform`（FHT）的正确性及性能对比；
- HadaCore 纯 Tensor Core 基线和实验性的 MMA/CUDA Core hybrid 实现；
- Hadamard + per-row/per-group INT8、INT4、FP8 动态量化融合。

项目不依赖 Python、PyTorch 或虚拟环境。核心 benchmark 由 `nvcc` 直接编译，输入使用 cuRAND 生成，kernel 延迟使用 CUDA event 测量。当前报告中的硬件与性能数据均来自本机 NVIDIA GeForce RTX 4050 Laptop GPU，不再使用此前 A800 的数据。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `hadamard.cu` | CUDA Core 实现：small、scalar、vec、multi-warp、chunked multi-warp 与 v4 自动 dispatch |
| `hadacore.cu` | 参考 HadaCore 改写的纯 Tensor Core 基线 |
| `hadamard_hybrid.cu` | 将 MMA 用于局部 H16、其余阶段使用 CUDA Core 的实验实现 |
| `bench.cu` | 普通 Hadamard 的正确性与性能 benchmark |
| `fht/` | 去除 Torch/C10 依赖后的 FHT CUDA 源码 |
| `fuse/fuse_per_row_quant.*` | Hadamard + per-row quant 融合实现 |
| `fuse/fuse_per_group_quant.*` | Hadamard + per-group quant 融合实现 |
| `fuse/fuse_bench.cu` | per-row 融合与“Hadamard v4 + 独立量化”的对比 benchmark |
| `fuse/fuse_group_bench.cu` | per-group 融合 benchmark |
| `plot_heatmaps.py` | small/scalar/vec 相对 FHT 的加速比矩阵和热力图 |
| `plot_v4_heatmap.py` | v4 相对 FHT 的加速比矩阵和热力图 |
| `prof.sh` | Nsight Compute 采集示例 |
| `docs/report.md` | 实现、优化过程、精度与最终性能报告 |

## 环境要求

- NVIDIA GPU 与驱动；
- 支持 C++17 的 CUDA toolkit，且 `nvcc` 位于 `PATH`；
- `libcurand`；
- fused FP8 路径需要 CUDA toolkit 提供 `cuda_fp8.h`。

Makefile 默认 `ARCH=sm_80`。应按实际 GPU 修改，例如本机 RTX 4050 使用：

```bash
make ARCH=sm_89 bench
```

也可以用 `NVCC=/path/to/nvcc` 指定编译器。

## 构建

```bash
make bench                 # 普通 Hadamard benchmark
make fuse_bench            # fused per-row quant benchmark
make fuse_group_bench      # fused per-group quant benchmark
make clean
```

## 普通 Hadamard benchmark

示例：

```bash
./bench --impl 4 --rows 1024,2048,4096 \
  --dims 256,512,1024,2048,4096,8192 --dtype all
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--impl` | `2` | `1`=scalar，`2`=vec，`3`=multi-warp，`4`=自动 dispatch，`hybrid`=MMA/CUDA Core 实验实现 |
| `--rows` | `small` | 行数列表；支持 `small`、`middle`、`large`、`all` 或逗号分隔整数 |
| `--dims` | `small` | COLS 列表，格式同 `--rows` |
| `--dtype` | `all` | `fp16`、`bf16` 或 `all` |
| `--iters` | `100` | 正式计时的启动次数 |
| `--warmup-ms` | `200` | 每个 case 正式计时前至少运行的时间 |
| `--seed` | `0` | 输入随机种子 |
| `--no-check` | 关闭 | 跳过正确性校验 |
| `--no-bench` | 关闭 | 跳过性能测试 |
| `--no-warmup` | 关闭 | 跳过 benchmark 开始前的全局 GPU 预热 |

v4 支持 `COLS=2...32768` 的 2 的幂，并按形状选择：

- `2...16`：一个 warp 同时处理多行；
- `32...128`：scalar one-warp-per-row；
- `256...8192`：小 rows 使用 multi-warp-per-row，其余使用 vec；
- `16384/32768`：chunked multi-warp-per-row。

`hybrid` 原生支持 `COLS=256...32768`；benchmark 在更小维度自动回退到 v4。

正确性参考为从存储类型输入扩展到 FP32 后执行的 CPU FWHT。项目门槛为：

- FP16：与 FHT 的最大绝对误差不超过 `1e-2`；
- BF16：与 FHT 的最大绝对误差不超过 `5e-2`。

FHT 支持到 `COLS=32768`；`COLS=2/4` 会零填充到 8 后计算并截取结果。CPU 参考校验为控制内存开销只运行到 `COLS=8192`，更大尺寸仍可计时。

注意：CLI 的 `large/all` 维度预设还包含 `65536`，而当前 v4、FHT 和融合实现的上限是 `32768`。测试这些实现时应显式传入不超过 `32768` 的 `--dims`。

HadaCore 会在其支持的形状上同时运行并输出 `hada/fht` 加速比。由于 MMA 改变运算顺序，且多个 MMA 阶段之间存在 FP32 到 FP16/BF16 的转换，HadaCore 和 hybrid 是实验性 Tensor Core 路径，部分形状可能无法通过上述严格误差门槛；这不代表 CUDA Core v4 的正确性失败。详细分析见 `docs/report.md`。

当前 `bench.sh` 是一个可编辑的单 case 快捷入口，默认运行 hybrid 的 `rows=2048, cols=2048`，不是全量测试脚本。

## Fused Hadamard + Quant

per-row 示例：

```bash
./fuse_bench --rows 32,2048 --dims 256,1024,4096,8192 \
  --dtype all --quant all --scheme all --policy auto \
  --iters 100 --warmup-ms 20
```

也可以运行 `bash fuse/bench.sh`。支持：

- INT8：symmetric / asymmetric；
- INT4：symmetric / asymmetric，两个量化值打包到一个 byte；
- FP8 E4M3：scale-only；
- `--policy auto|vec|multi|all`，用于对比 vec 与 multi-warp dispatch。

benchmark 将融合结果与以下非融合路径逐字节比较，并报告性能加速比：

```text
Hadamard v4 -> FP16/BF16 中间结果 -> standalone quant
```

per-group 示例：

```bash
./fuse_group_bench --rows 2048 --cols 8192 --group-size 128 \
  --dtype all --quant all --scheme all --iters 100 --warmup-ms 20
```

也可以运行 `bash fuse/group_bench.sh`。`cols` 与 `group-size` 必须是 2 的幂，且 group size 必须整除 cols。当前未实现 per-tensor 融合量化，因为它需要跨 CTA 的全局 min/max reduction 和额外同步或第二阶段 kernel。

## 热力图与 NCU

热力图脚本需要已有的 `./bench`，以及当前 Conda 环境中的 NumPy 和 Matplotlib：

```bash
python plot_heatmaps.py
python plot_v4_heatmap.py
```

两者都会在终端打印具体加速比矩阵，并输出 `heatmap_*.png`。

采集 vec 与 FHT 的 Nsight Compute 报告：

```bash
bash prof.sh 2 2048 8192 fp16
```

可通过 `ROWS`、`DIMS`、`DTYPE`、`SET`、`OUT` 和 `NCU_ARGS` 环境变量调整采集配置。
