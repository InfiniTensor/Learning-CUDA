# CUDA Core FWHT version benchmark

本项目优化归一化 Walsh-Hadamard Transform：

```text
y = x @ H(D) / sqrt(D)
```

自研版本、Dao-AILab `fast-hadamard-transform` 都使用相同的 Sylvester/Walsh 顺序与归一化。`torch.clone` 只表示显存读写下界，不参与数学正确性比较。

## 版本

| 版本 | 文件 | 策略 |
|---|---|---|
| `hadamard_v0` | `kernels/hadamard_v0.cu` | global memory、每个 FWHT stage 一次 launch |
| `hadamard_v1` | `kernels/hadamard_v1.cu` | 每行一个 block、FP32 shared memory |
| `hadamard_v2` | `kernels/hadamard_v2.cu` | 向量化 I/O、warp shuffle、shared memory |

版本一经记录不再修改。下一次优化复制 `hadamard_v2.cu` 为 `hadamard_v3.cu`，注册到枚举、CMake 和 Python `VERSION_IDS` 后，会自动进入 correctness、benchmark 和报告。

## 一键运行

```bash
cd /home/czw1/infinitensor/Hadamard/hadamard_cudacore
./scripts/bootstrap_env.sh       # 首次安装 PyTorch + Dao FHT
./scripts/run_benchmark.sh       # 12 个代表性 shape
./scripts/run_benchmark.sh --full # 5 batch × 7 D 完整矩阵
```

默认矩阵测试 FP32/FP16/BF16；固定 `batch=32` 扫全部 D、固定 `D=1024` 扫全部 batch，并增加 `[4096,8192]` 压力点。每个实现独立跑 3 轮，单轮 warmup 100、repeat 1000，最终使用中位数。

每次运行输出到 `results/<timestamp>_representative/`：

- `metadata.json`：GPU、CUDA、Torch、Dao commit、参数；
- `same_process_raw.csv` / `native_raw.csv`：三轮原始数据；
- `same_process_summary.csv` / `native_summary.csv` / `summary.csv`：中位数、波动和速度比；
- `report.md`：按 dtype/shape 的可读表格、相对 Dao 的几何平均速度和不稳定 case。

`results/latest` 指向最近一次结果。

## 指标与数学门禁

每个版本均用随机、单位基向量、确定性序列校验：`version vs reference`、`Dao vs reference`、`version vs Dao`。任意一项失败，该版本标记 `COMPARISON INVALID`，但仍保留原始 latency。

- `speedup_vs_v0 = v0_latency / current_latency`；大于 1 表示比 v0 快。
- `speed_vs_dao = dao_latency / current_latency`；大于 1 表示当前版本比 Dao 快。
- `latency_ratio_vs_dao = current_latency / dao_latency`；大于 1 表示更慢。
- `percent_latency_gap = (current/dao - 1) * 100%`；正数表示更慢。
- `ratio_vs_clone = current_latency / clone_latency`。

`v0` 每一级都以输入 dtype 回写；因此 BF16/D=8192 使用单独记录的 `atol=rtol=0.08`。v1/v2 仍使用更严格的 FP32 中间累加容差。

## 原生与 NCU

原生 benchmark 使用 `cudaEvent_t`，不计入分配和 memcpy：

```bash
./build/benchmark_cuda --kernel v2 --batch 1024 --dim 4096 \
  --dtype fp16 --warmup 100 --repeat 1000
```

旧参数 `naive/shared/optimized` 暂时兼容，但输出统一为 `hadamard_v0/v1/v2`。

NCU profile 模式仅正式执行 3 次：

```bash
ncu --profile-from-start off --set full --launch-count 1 \
  ./build/benchmark_cuda --kernel v2 --batch 1024 --dim 4096 \
  --dtype fp16 --profile
```

Python 侧可将 `--implementation` 改为 `v0`、`v1`、`v2`、`dao` 或 `clone`。NCU 下的 Event latency 会受插桩影响，仅用于看 kernel 指标，不用于正常性能结论。

```bash
ncu --profile-from-start off --set full --launch-count 1 \
  .venv/bin/python benchmark/benchmark_fht.py \
  --implementation v2 --batch 1024 --dim 4096 --dtype fp16 --profile
```
