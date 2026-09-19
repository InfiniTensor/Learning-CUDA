# MXFP8 / NVFP4 CUDA Quantize-Dequantize

这是一个不依赖原生 FP8/FP4 指令或第三方量化库的软件模拟项目。程序读取
FP16/FP32 `QDAT` 矩阵，在 GPU 上量化为 packed MXFP8 或 NVFP4，写出带校验的
`QLOW` 文件，再从该文件重新读取并在 GPU 上反量化为 FP16、BF16 或 FP32。

详细设计、公式、二进制协议、逐文件/逐函数说明和实验结果见
[REPORT.md](REPORT.md)，原始规划见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 快速开始

```bash
./scripts/bootstrap.sh
./scripts/run_correctness.sh
./scripts/run_benchmarks.sh
```

也可以手动执行一条流水线：

```bash
python3 tools/generate_test_data.py generate \
  --output data/demo.qdat --kind normal --rows 64 --cols 65 --dtype fp32

./build/quant_dequant \
  --input data/demo.qdat \
  --config configs/mxfp8_block.toml \
  --quantized results/demo.qlow \
  --output results/demo_dequant.qdat \
  --metrics results/demo.json \
  --warmup 10 --iterations 100

python3 tools/generate_test_data.py inspect results/demo_dequant.qdat
```

`quant_dequant` 默认同时运行 CPU reference，并要求 CPU/GPU 的量化字节、scale
和 FP32 反量化结果完全相同。大规模纯性能实验可显式加入
`--no-reference-check`，但正确性验收不应关闭它。

## 输出

- `*.qlow`：自描述、16 字节分段对齐、带 FNV-1a 校验的低精度文件；
- `*.qdat`：FP16/BF16/FP32 反量化矩阵；
- `*.json`：误差、压缩率、CUDA event kernel 时间和有效带宽；
- `results/benchmarks/summary.csv`：所有 benchmark JSON 的汇总表。

## 当前范围

第一版实现 nearest ties-to-even、基于固定 seed 的 stochastic rounding、block/tensor
scaling、尾块、奇数 FP4 打包和 NaN/Inf 的确定性处理。当前 benchmark 报告 CUDA event
的多次平均值，不是 p10/median/p90。详见报告的限制章节。
