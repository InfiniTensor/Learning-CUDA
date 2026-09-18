# records：实验过程证据

保存每次优化的**原始证据**：benchmark JSONL、环境信息、源码哈希、nsys 报告、验证日志。

本目录用于审计与复现，不是查找交付输出的地方——三类交付文件在 `output/`。

---

## 目录作用

| 目录 | 作用 |
|---|---|
| `01-before`～`12-*` | CUDA 各优化轮次的 benchmark、验证与 profile |
| `musa-opt01`、`musa-opt02`、`musa-s4000-20260915` | 摩尔线程平台的适配与优化 |
| `controlled-v1` | 固定输入、跨平台（CUDA/MUSA）的复测基线 |
| `benchmark-audit` | 测试条件、输入哈希与审计清单 |
| `_archive` | 旧版、重复实验与历史快照 |
| `08-nvfp4-float2-test` | 已放弃方案的负结果记录 |

各轮目录中的 `benchmark/`、`profile/`、`validation/`、`repeat/` 分别表示性能、nsys、正确性与复测证据。

---

## 单轮目录包含什么

一次正式的优化轮次（如 `records/11-nvfp4-partials1024/`）通常包含：

```
<轮次>/
├── RESULTS.md                可读的性能汇总与结论
├── summary.json              结构化汇总
├── <格式>_<元素数>.jsonl     原始每次测量
├── environment.json          GPU、驱动、编译器版本与源码哈希
└── repeat/                   复测，用于确认结果稳定
```

`environment.json` 里的 `source_sha256` 记录了当轮源码的哈希，用于确认测量对应的确切代码版本。

---

## 命名与规则

- 新实验**必须使用新目录名**，禁止覆盖旧记录
- 目录名体现轮次与改动内容，如 `07-mxfp8-vectorized`、`11-nvfp4-partials1024`
- 生成记录的命令**同一目录名只能运行一次**；目录已存在会报 `FileExistsError`，属有意设计
- 重新运行时把名字递增，例如 `my-benchmark-01` → `my-benchmark-02`
- `_archive/` 存放不再作为当前结论依据的历史数据
- 每轮数据与 `OPTIMIZATION_LOG.md` 中的条目对应

---

## 生成方式

### 性能证据

```bash
python3 Core/tools/benchmark.py --directory records/my-benchmark-01 --repeats 20
```

产物：`RESULTS.md`、`summary.json`、`<格式>_<元素数>.jsonl`、`environment.json`。

`--directory` 必须是**未存在**的新目录，不覆盖旧结果。

### nsys 证据

```bash
python3 Core/tools/profile.py --directory records/my-profile-01
```

产物：`mxfp8.nsys-rep`、`nvfp4.nsys-rep`、对应的 `.sqlite` 与 `*_capture.log`、`*_stats.txt`、`version.txt`。

### 误差评估输出

```bash
python3 Core/tools/quantize.py evaluate --backend cuda
```

---

输入复用 `input/evaluation-v1/`，评估的权重、张量与统计保存到 `output/evaluation-v1/cuda/`（重复运行自动加编号）。历史评估证据仍保留在原目录；新的交付输出不再写入records。

## 相关文档

- 性能口径与加速比规则见 [固定协议](../docs/BENCHMARK_PROTOCOL.md)
- 交付输出见 [output/README.md](../output/README.md)
- 本地 ncu/nsys 报告见 [results/README.md](../results/README.md)
