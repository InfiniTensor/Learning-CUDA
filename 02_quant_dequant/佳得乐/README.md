# MXFP8 / NVFP4 CUDA 软件模拟

完整支持 MXFP8-E4M3、MXFP8-E5M2、NVFP4-E2M1；CPU C++17 reference 以及 CUDA baseline / optimized 两个版本。输入 FP32/FP16，输出 FP32/FP16/BF16，block/tensor，nearest/stochastic。RTX 3090 sm_86 为实测目标。其他硬件未实测。

## 快速运行

从本目录运行。依赖 CMake ≥3.16、C++17、CUDA Toolkit（实测12.6）、Python3 与 matplotlib（实测3.9.2）。不用原生 FP8/FP4 指令。

```bash
python3 scripts/reproduce.py quick
python3 scripts/reproduce.py full
```

每次新建 `results/<run_id>/`，同名目录拒绝覆盖。脚本完成构建→标量/CPU/GPU测试→实验→汇总→SVG/PNG；full 另含大矩阵、额外分布及 sanitizer/profiler。失败返回非零；ncu/nsys 的可选采集失败单独记录，不冒充通过。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86
cmake --build build -j 4
(cd build && ctest --output-on-failure)
mkdir -p results/manual
./build/lp generate --rows 3 --cols 17 --dtype fp16 --distribution normal --seed 42 --output results/manual/input.tensor
./build/lp quantize --input results/manual/input.tensor --config configs/nvfp4.toml --backend cuda --kernel_variant optimized --output results/manual/weights.lp
./build/lp dequantize --input results/manual/weights.lp --backend cuda --output results/manual/restored.tensor
./build/lp verify --input results/manual/input.tensor --packed results/manual/weights.lp
./build/lp benchmark --input results/manual/input.tensor --config configs/nvfp4.toml --kernel_variant optimized --warmup 3 --repeat 5 --workdir results/manual --csv results/manual.csv
```

MX 两条路径分别使用 `configs/e4m3.toml`、`configs/e5m2.toml`。配置支持的字段以这些文件为准，未知/重复/不适用字段报错；NV 禁止 fp8_type。只支持默认块大小 32/16。tensor 与尾块在文件元数据标为扩展。

`quantize` 和 `dequantize` 均可选择 `--backend cpu`。CPU 独立构建：

```bash
cmake -S . -B build-cpu -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
(cd build-cpu && ctest --output-on-failure)
```

`verify --backend cpu` 可用于无 GPU 环境。`benchmark` 专用于 GPU 与 CPU 对照，需要 CUDA。

## 阅读与证据

- [数值规则与来源](docs/numerics.md)、[文件格式](docs/file_format.md)。
- [报告](docs/report.md)、[逐条验收](docs/requirements_matrix.md)、[学习指南](docs/learning_guide.md)。
- 全部实验保存 seed、源代码摘要、配置、环境、构建缓存、原始 CSV、命令和日志。图表由 `scripts/summarize.py` 自动生成。
- CPU 使用单线程 `-O3`；报告区分纯 kernel、含传输 API 流程和含文件 I/O 流程。小矩阵不保证 GPU 更快。

## 测试说明

项目里保留了独立的 tests/ 目录。选题二明确要求提交“程序（包含测试）”，所以这里以选题要求为准

## 清理

构建产物、测试临时文件和实验结果都放在忽略目录里，不会混进源码。清理时用 scripts/clean.py：
    scripts/clean.py --run-id <已存在实验ID>：只删掉带 metadata.json 的那个实验目录；
    scripts/clean.py --build：只删本项目的 build/ 和 build-cpu/。
