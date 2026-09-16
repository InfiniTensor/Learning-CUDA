# 稳定性矩阵与实现选择规则

## 第八阶段覆盖

`stability_tests` 使用可被 FP16 精确表示的确定性数据，并为数据库加入重复向量、
为查询加入全零向量，使相同分数必须按较小 ID 排序。GPU 结果按候选 ID、名次和
score 与 CPU Exact/IVF 参考比较，而不是只检查结果数量。

| 类别 | 覆盖 |
|---|---|
| Exact 主矩阵 | 64/128/384/768 维 × batch 1/8/32/40/64/128 × K 1/10/50/100 × FP32/FP16 × L2/IP/cosine，共 576 组 |
| batch 边界 | 65 维，每个 batch 使用 `batch+1` 个 query，覆盖一个满 batch 和尾 batch |
| 非整 warp 维度 | 65 维 × FP32/FP16 × L2/IP/cosine，K=100 |
| Exact 推荐路径 | 128/K10/tile4/64 blocks、384/K10/tile8/160 blocks、384/K50/tile4/64 blocks |
| Exact 安全回退 | 不支持的 FP32 768 维 block-tile 与 K50/tile8 均 fail closed；改用 warp-query/tile1 后通过 |
| IVF 病理桶 | 65 维，8 桶中 4 个空桶、最大桶 125/128；K 1/10/50/100 × FP32/FP16 × 三 metric × warp/bucket-major，共 48 组 |

原始摘要位于 `../results/stage8_stability/coverage.csv`。远端完整 CTest 14/14
通过；`compute-sanitizer --tool memcheck` 对整个 `stability_tests` 报告
`ERROR SUMMARY: 0 errors`。

## Exact 选择规则

下表中的优化项只适用于已有正式 A/B 的 RTX 4090 D、FP32、batch=64 组合。
规则刻意保守：正确性测试通过不等于在未测组合上已有性能证据。

| K | 维度/metric | 推荐 layout | query tile / blocks | 其他情况 |
|---:|---|---|---|---|
| 10 | 128，L2 或 inner product | warp-query | 4 / 64 | 回退 1 / 0 |
| 10 | 384，cosine | warp-query | 8 / 160 | 回退 1 / 0 |
| 50 | 384，cosine | warp-query | 4 / 64 | 回退 1 / 0 |
| 1、100 | 任意已测维度 | warp-query | 1 / 0 | 同左 |
| 任意 | 64、768、FP16、非 batch64 或其他 GPU | warp-query | 1 / 0 | 同左 |

其中 `blocks=0` 表示使用稳定的并行度启发式。`block_tile` 在 Synthetic/Yahoo
均未超过 warp-query，不进入任何推荐组合。显式请求不支持的实验组合时，参数
验证会拒绝启动 kernel；调用方必须改回
`exact_fused_layout=warp_query, exact_query_tile=1, exact_fused_blocks=0`。
这是一种 fail-closed 回退，不会静默运行未经验证的近似路径。

## IVF 选择与回退

- K=10、固定 nprobe：使用完整候选的 `bucket_major`。
- K=50/100：未做数据集级 recall 验证时使用完整 heap 的 `bucket_major`。
- 分区 K=50/100 只作为候选运行。`scripts/run_real_benchmarks.py` 会将其与完整
  heap 的全部 nprobe 点比较；任一点 recall 损失超过 `1e-4`，最终 CSV 自动
  回退到 `*_safe.conf` 的完整 heap 结果。
- 自适应 nprobe 或 memory scoring 不使用分区模式；参数验证会拒绝不兼容组合。
- 空桶、严重倾斜桶、重复向量、相同分数和 65 维均已验证完整 heap 路径；这些
  场景不会自动切换到分区候选。

## 适用边界

本阶段证明的是单卡、内存内、K≤100、维度≤768 的已列组合正确且无 memcheck
错误，不代表其他 GPU 上的最优性能选择。切换 GPU 架构、batch、dtype 或数据
分布时，先使用安全 baseline，再按第六/七阶段的严格 ID/score 或 recall 门槛
重新接纳优化配置。
