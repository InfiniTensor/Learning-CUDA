# results：本地原始分析报告

保存 nsys / ncu 的**原始报告与结论**，用于写总结报告时查阅。

本目录体积较大，**报告文件不入库**（`.ncu-rep`、`.nsys-rep` 已被 `.gitignore` 排除），仅保留 `.md` 结论与索引。

---

## 目录内容

```text
results/ncu/
├── README.md                    索引：哪个报告对应哪个 kernel
├── mxfp8-test-analysis.md       MXFP8 的 ncu 结论
├── nvfp4-test-analysis.md       NVFP4 的 ncu 结论
├── mxfp8-test.ncu-rep           ncu 原始报告（本地保留，不入库）
├── nvfp4-test.ncu-rep
├── nvfp4-maximum.ncu-rep
└── nvfp4-maximum-1024.ncu-rep
```

---

## 打开原始报告

```bash
ncu --import results/ncu/mxfp8-test.ncu-rep --page details
ncu --import results/ncu/nvfp4-test.ncu-rep --page details
```

导出为 CSV：

```bash
ncu --import results/ncu/mxfp8-test.ncu-rep --page details --csv
```

CSV 的列为 `Kernel Name`、`Metric Name`、`Metric Value`、`Duration`。

采集新报告：

```bash
ncu --set detailed -k regex:mxfp8_quantize ./Core/build/pipeline_mxfp8 --benchmark 4194304 20
```

---

## 与 records/ 的区别

| 目录 | 内容 |
|---|---|
| `results/` | 本地的 nsys/ncu 原始报告与分析结论，体积大，不入库 |
| `records/` | 每次实验的完整证据链（JSONL、环境、哈希、日志），入库 |

按阶段定位用 nsys（见 `Core/tools/profile.py`），看 kernel 内部指标用 ncu。

---

## 相关文档

- 索引与文件名对应关系：[ncu/README.md](ncu/README.md)
- nsys 采集记录：[records/09-nsys-mxfp8/](../records/09-nsys-mxfp8/)
