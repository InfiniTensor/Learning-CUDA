# 第七阶段证据索引

- `gpu.txt`：正式实验 GPU 型号、驱动和显存。
- `source.sha256`：首次正式分段剖析时的二进制、CUDA 源码和驱动脚本校验和。
- `current_source.sha256`：加严逐 rank 校验并撤回负优化后的远端二进制、CUDA
  源码和驱动脚本校验和；恢复任务状态为 0，CTest 12/12 通过。
- `cosine_norm_ab_source.sha256`：范数预计算 A/B 的基线/候选二进制、候选源码
  与重放脚本校验和。
- `run_exact_profile_ab.sh`：在远端 `/data` 项目目录重放四组严格 A/B 的脚本。
- `query_tile_source.sha256`：query-tiled fused 最终二进制、源码、驱动和保留
  配置的校验和，以及 CTest/Sanitizer 状态。
- `block_tile_source.sha256`：block-tile 二进制、源码、配置、两份 smoke CSV
  的校验和，以及 CTest/Sanitizer 状态。
- `../../../results/stage7_exact_profile/`：Synthetic-1M 与 Yahoo MiniLM 384
  分段基线 CSV，以及范数预计算的四份 A/B CSV。
- `../../../results/stage7_exact_query_tile/`：两轮 smoke、grid.x sweep，以及
  Synthetic/SIFT/Yahoo 的 20 次正式 A/B CSV。
- `../../../results/stage7_exact_block_tile/smoke/`：Synthetic/Yahoo 的
  block-tile grid.x 筛选 CSV；该候选未超过 warp-query，未扩展正式长跑。

分段方法是 differential estimate：distance-only kernel 保留 fused 路径的
网格、query cache 和距离归约，局部选择估算为两者差值；final merge 独立计时。
首次正式基线做候选集合与分数校验；随后 A/B 复测加严为按 query、rank、ID
和分数比较。长任务使用 `nohup`，原始输出和日志仅保存在远端
`/data/purplepulse-work/`。

query-tile 长任务目录以 `stage7-query-tile-` 开头。最终构建 CTest 13/13
通过，`stage7-query-tile-sanitizer-20260910` 状态为 0，Compute Sanitizer
报告 0 errors。大日志和 Top-K 文本输出只保存在远端 `/data`。

block-tile 使用 `stage7-block-tile-build-20260910` 构建并通过 CTest 13/13；
筛选任务为 `stage7-block-tile-smoke-20260910`。默认 layout 仍是
`warp_query`，block-tile 仅在配置显式请求时启用。
`stage7-block-tile-sanitizer-20260910` 状态为 0，Compute Sanitizer 报告
0 errors。
