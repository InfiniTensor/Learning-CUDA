# CUDA Hadamard 与融合动态 INT8 量化

本目录是 `czwdysj` 的 2026 夏季训练营 CUDA Hadamard 项目提交，包含两个可独立构建、测试和分析的工程：

- [`hadamard_cudacore`](hadamard_cudacore)：归一化 Fast Walsh-Hadamard Transform（FWHT）的 v0/v1/v2 实现。
- [`Quantification_hadamard_cudacore`](Quantification_hadamard_cudacore)：FWHT、逐行 `amax` 和动态对称 INT8 量化的 v0/v1/v2 融合实现。

## 数学定义

两个工程使用相同的 Sylvester/Walsh Hadamard 排列和归一化：

```text
y = x @ H(D) / sqrt(D)
```

融合工程继续计算：

```text
amax[row]  = max(abs(y[row, :]))
scale[row] = amax[row] / 127                 (全零行使用 scale = 1)
q          = round(clamp(y / scale, -127, 127)).to(int8)
```

性能基线为数学等价的 `Dao-AILab/fast-hadamard-transform + PyTorch eager/torch.compile` 量化链。`torch.clone` 只用作显存搬运参考，不作为等价算子。

## 版本说明

| 工程 | v0 | v1 | v2 |
| --- | --- | --- | --- |
| FWHT | global-memory 多 stage 基线 | shared-memory FWHT | warp shuffle + 向量化访存 |
| 融合算子 | shared FWHT + shared reduction | 向量化访问 + warp reduction | warp shuffle FWHT + 打包 INT8 写回 |

详细约束、接口和新增版本方式见两个子目录的 `README.md` 与 `VERSIONING.md`。

## 构建与快速验证

```bash
cd 03_hadamard_tc/czwdysj/hadamard_cudacore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

cd ../Quantification_hadamard_cudacore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

安装固定的 PyTorch/Dao FHT 环境并运行对比：

```bash
cd ../hadamard_cudacore
./scripts/bootstrap_env.sh
./scripts/run_benchmark.sh

cd ../Quantification_hadamard_cudacore
./scripts/run_pytorch_comparison.sh
```

完整矩阵可追加 `--full`。NCU 单 case、批量采集和 NCU-UI 报告复现命令见融合工程 README。

## 已提交结果

- [`report/REPORT.md`](report/REPORT.md)：RTX 5060 Laptop 上的正确性、延迟、性能比和瓶颈分析。
- [`results/summary.csv`](results/summary.csv)：代表性 shape/dtype 的汇总结果。
- [`results/correctness.csv`](results/correctness.csv)：完整交叉正确性门禁。
- [`results/metadata.json`](results/metadata.json)：GPU、CUDA、PyTorch 和测试参数。

代表性运行覆盖 FP32/FP16/BF16 共 36 个 shape/dtype case；216 条正确性记录全部通过。融合 v2 相对 PyTorch eager 的几何平均加速为 9.054x，相对 `torch.compile` 路径为 4.833x。原始大体积结果和 `.ncu-rep` 未提交；仓库保留了可复现它们的脚本，避免将构建产物、虚拟环境和分析二进制塞入 Git。

