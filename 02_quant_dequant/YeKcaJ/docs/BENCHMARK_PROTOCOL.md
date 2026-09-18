# 性能测试固定协议

## 只用这一组条件

| 项目 | 固定值 |
|---|---|
| 输入 | FP32，形状1×N；正态 N(0,1)，生成 seed=20260909 |
| 元素数 | 1,048,576 / 4,194,304 / 16,777,216 |
| 文件 | `input/benchmark-v1/normal_N.fp32`，生成一次后冻结；SHA256 必须匹配 `Core/configs/benchmark-inputs-v1.json` 和本地 manifest.json |
| 格式 | MXFP8 / NVFP4 分开记录；block size分别32 / 16 |
| 模式 | block / nearest；舍入 seed=1234（nearest不使用随机数） |
| 计时 | 预热3次，测量20次，复用显存，GPU event |
| 主指标 | `quant_optimized / resident_gpu`，median ms与P95 ms |
| 统计 | 偶数样本中间两值均值；P95 nearest-rank |
| 排除 | CPU校验、文件IO、分配、传输、释放；NVFP4全局归约必须包含 |
| 反量化 | FP32/FP16/BF16单独统计，不能和量化混算 |

主机 workspace 复用、nsys/ncu/MUPTI、算术微基准、单次 CLI 都是独立诊断数据，不填入主表。GPU block/grid、融合和编码实现可以作为优化变量，每轮明确只改什么；涉及多项变更时不能把合并收益归因于其中一项。

## 运行

```bash
# CUDA，先构建当前源码
cmake --build Core/build -j 4
python3 Core/tools/benchmark.py --backend cuda --directory records/my-next-cuda

# MUSA，先构建当前源码
cmake --build Core/build-musa -j 4
python3 Core/tools/benchmark.py --backend musa --directory records/my-next-musa
```

首次运行会导出旧 benchmark 的 FP32 正态输入并保存 SHA256 清单；以后复用文件，哈希改变或缺失即拒绝测试。两平台必须使用相同清单/文件：即使 seed相同，标准库实现不同也可能生成不同输入。跨平台报告脚本会核对实际 SHA256，不同就拒绝比较，需复制同一套输入再测。

`--repeats` 正式固定20；非20次需显式 `--exploratory`，这种结果不允许进入正式跨平台对比。直接运行旧式 `pipeline --benchmark N R` 仍可诊断，但没有冻结文件/元数据，不当正式记录。旧二进制不支持新增的文件参数时，先按相同算法版本重建，不静默退回旧入口。

### 冻结输入不在版本库里

`input/**/*.fp32` 被 `.gitignore` 排除，仓库只保留 `manifest.json` 和
`Core/configs/benchmark-inputs-v1.json` 两个哈希清单。因此**新克隆的仓库直接跑
`benchmark.py` 会报 `fixed input missing`，这是预期行为**，不是脚本坏了。

恢复方式：用同一份源码重建二进制，执行导出命令。导出的文件由固定 seed 生成，
实测与冻结文件逐字节一致，SHA256 与两个清单相符后才会开始计时。

```bash
cmake --build Core/build -j 4
mkdir -p input/benchmark-v1
for n in 1048576 4194304 16777216; do
  Core/build/pipeline_mxfp8 --benchmark-export "$n" "input/benchmark-v1/normal_$n.fp32"
done
```

导出后核对（应分别等于两个清单里的值）：

```bash
sha256sum input/benchmark-v1/normal_*.fp32
```

`benchmark.py` 会在计时前后各校验一次哈希，缺失或被改动都会拒绝测试。若哈希与清单
不符，说明源码中的输入生成逻辑已被改动，此时不能用新文件冒充 v1 数据，应新建语料版本。

每份报告保存完整输入参数、输入/二进制/源码 SHA256、设备/编译器、所有命令返回的 JSONL。GPU任务串行执行；接电、关闭其他GPU任务、保持同一功耗设置，记下额外环境变化。当前未锁频，因此不声称硬件状态绝对一致。

## 加速比只有两列

- 相对上一轮 = 上一轮**已保存结果** median / 本轮 median。
- 相对基线 = 该平台该格式**优化前基线** median / 本轮 median（同元素数）。

CUDA 历史日志保留原第0～6次编号，基线统一为 `records/01-before/benchmark/summary.json` 的第0次数据。MUSA 单独记录，基线为 `records/musa-opt01/baseline/summary.json` 的第0轮数据。各格式首次优化的上一轮就是基线。before复测不替换已保存的上一轮或基线；不从不同运行里各挑一行最好成绩。各日志末尾只汇总各版本相对基线的加速比。

旧历史记录缺少实际输入哈希，单列历史趋势。新的严格协议从当前 `records/controlled-v1/` 建立独立可复现基线，不能假装是把六次历史版本重测了一遍。

生成历史表：`python3 Core/tools/optimization_report.py`。

严格跨平台比较使用 `Core/tools/compare_benchmarks.py`，报告直接保存到指定的 `records/controlled-v1/` 目录；硬件不同，因此跨平台耗时比独立于优化链，不能写成算法优化加速比，也不用历史最低值冒充当前版本。
