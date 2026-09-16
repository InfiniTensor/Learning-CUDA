# 2026-09-10 第六阶段公开数据命令

所有长任务在实验机项目目录执行并用 `nohup` 脱离 SSH。登录地址和凭据不记录
在仓库中。原始数据、索引、Top-K 输出和日志保存在 `/data`；仓库只同步 CSV、
JSON 和 SVG 摘要。

## 现有产物检查

启动 Yahoo 前先检查进程和两套结果目录。SIFT1M 已包含 `build.csv`、
`exact_gpu.csv`、K=10/50/100 的完整 IVF 矩阵、21 行 FAISS 数据、ground truth
验证和图表，因此没有重复运行。Yahoo 数据已转换但结果目录为空。

```bash
pgrep -af 'prepare_real_datasets|run_real_benchmarks|ivf_build|ivf_search|vector_search|benchmark_faiss'
find results/real/sift1m -maxdepth 2 -type f -print
find results/real/yahoo-minilm-384 -maxdepth 2 -type f -print
sha256sum data/real/yahoo-minilm-384-normalized/manifest.json
```

Yahoo manifest 记录原始 HDF5 SHA-256 为
`8b0519850249d7ea4aea258eb43ae837dca1bb2f7a85ee0a5145afdb699c4136`，转换结果为
677,305 × 384、1,000 queries、cosine、Top-100 ground truth。

## Yahoo MiniLM 384 正式运行

任务信息写入 `/data/purplepulse-work/jobs/stage6-yahoo-minilm-384-20260910/`，结果
写入 `/data/purplepulse-work/PurplePulse/results/real/yahoo-minilm-384/`。

```bash
cd /data/purplepulse-work/PurplePulse
job_dir=/data/purplepulse-work/jobs/stage6-yahoo-minilm-384-20260910
mkdir -p "$job_dir"

nohup env \
  PYTHONUNBUFFERED=1 \
  PATH=/usr/local/cuda/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1 \
  /data/purplepulse-faiss-system-venv/bin/python \
  scripts/run_real_benchmarks.py \
  --dataset yahoo-minilm-384 \
  --project-root /data/purplepulse-work/PurplePulse \
  --data-root data/real \
  --results-root results/real \
  --build-dir build \
  --nlist 1024 \
  --nprobes 32,64,128,160,192,224 \
  --training-samples 100000 \
  --iterations 15 \
  --threads 16 \
  --batch-size 64 \
  --warmup 1 \
  --repeat 5 \
  > "$job_dir/driver.log" 2>&1 < /dev/null &

printf '%s\n' "$!" > "$job_dir/driver.pid"
```

驱动于 2026-09-10 11:55 UTC 完成。阶段结果由各 benchmark 脚本逐行更新 CSV，
即使 SSH 断开也不依赖终端缓冲。完整结果摘要与回退结论见
`docs/REPRODUCIBILITY.md`，可提交证据位于 `results/real/`。

## 第七阶段 Exact fused 剖析

长任务仍通过 `nohup` 运行，状态、PID 和日志均位于 `/data`。正式剖析任务为
`/data/purplepulse-work/jobs/stage7-exact-profile-formal-20260910/`，结果逐 K
checkpoint 到 `results/stage7_exact_profile/`。命令模板为：

```bash
cd /data/purplepulse-work/PurplePulse
job_dir=/data/purplepulse-work/jobs/stage7-exact-profile-formal-20260910
mkdir -p "$job_dir"
nohup bash -lc '
  cd /data/purplepulse-work/PurplePulse
  python3 scripts/benchmark_exact_breakdown.py \
    --search-binary ./build/vector_search \
    --database data/million/inner_fp32_db.bin \
    --queries data/million/inner_fp32_q.bin \
    --configs 10:configs/exact_million_k10_fused_batch64.conf,50:configs/exact_million_k50_fused_batch64.conf,100:configs/exact_million_k100_fused_batch64.conf \
    --reference-results 10:results/new_gpu/exact_k10.txt,50:results/new_gpu/exact_k50.txt,100:results/new_gpu/exact_k100.txt \
    --output-dir results/stage7_exact_profile/synthetic1m/outputs \
    --csv results/stage7_exact_profile/synthetic1m/breakdown.csv \
    --warmup 5 --repeat 20 --profile-repeat 20
' > "$job_dir/driver.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$job_dir/driver.pid"
```

Yahoo 使用 `data/real/yahoo-minilm-384-normalized/{database,queries}.bin`、
`configs/real_exact_k{10,50,100}.conf` 和第六阶段 Exact 输出作为逐 rank 参考。
首项 cosine 范数预计算 A/B 的完整可执行脚本为
`stage7/run_exact_profile_ab.sh`；对应 nohup 任务目录是
`/data/purplepulse-work/jobs/stage7-cosine-norm-ab-20260910/`。实验否决并恢复
baseline 后，`stage7-restore-profiler-baseline-20260910` 状态为 0，CTest
12/12 通过。

## 第七阶段 query tiling

query-tile A/B 使用 `scripts/benchmark_exact_query_tile.py`。驱动接受多个
`--variant label=K:config`，每完成一个 variant 就重写 checkpoint CSV，并用
`--reference-results` 做逐 query、逐 rank 的 ID/score 校验。远端长任务全部
通过 `nohup` 运行，主要目录如下：

- `stage7-query-tile-formal-20260910`：Synthetic/Yahoo K=10、K=50 正式 A/B。
- `stage7-query-tile-sift-formal-20260910`：只复测既有 SIFT1M 的 K=10/K=50。
- `stage7-query-tile-grid-sweep-smoke-20260910`：固定 `grid.x` 的短筛选。
- `stage7-query-tile-yahoo-grid-formal-20260910`：Yahoo K=10 的 20 次正式
  96/160/192-block 对照。
- `stage7-query-tile-sanitizer-20260910`：`compute-sanitizer --tool memcheck`
  执行 `unit_tests`，状态 0、0 errors。

所有 CSV 写入
`/data/purplepulse-work/PurplePulse/results/stage7_exact_query_tile/`；仓库只同步
CSV，不同步大型 Top-K 输出和日志。

## 第七阶段 block-tile 负实验

源码上传与构建任务分别位于 `stage7-block-tile-upload-20260910` 和
`stage7-block-tile-build-20260910`；上传前保存了 query-tile 最佳二进制，
构建状态为 0，CTest 13/13。筛选任务 `stage7-block-tile-smoke-20260910`
通过 `nohup` 运行 Synthetic/Yahoo K=10 的 baseline、warp-query 最优项和
block-tile 64/96/128/160/192 blocks。结果逐项 checkpoint 到
`results/stage7_exact_block_tile/smoke/{synthetic,yahoo}/results.csv`，全部通过
逐 query、逐 rank 的 ID/score 校验。候选未超过 warp-query，因此没有重复
SIFT 已完成工作，也没有启动 20 次正式长跑。Compute Sanitizer 任务目录为
`stage7-block-tile-sanitizer-20260910`。

## 第八阶段稳定性矩阵

上传任务目录为 `stage8-stability-upload-20260910`，其中保存上传前 CMake、测试
文件（若存在）和 `vector_search` 二进制。构建与完整回归通过以下 nohup 任务
执行，日志和状态均位于 `/data`：

```bash
cd /data/purplepulse-work/PurplePulse
job_dir=/data/purplepulse-work/jobs/stage8-stability-build-20260910
nohup bash -lc '
  cd /data/purplepulse-work/PurplePulse
  export PATH=/usr/local/cuda/bin:$PATH
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native &&
  cmake --build build &&
  ctest --test-dir build --output-on-failure
' > "$job_dir/driver.log" 2>&1 < /dev/null &
```

CTest 14/14 通过。`stage8-stability-sanitizer-20260910` 先执行
`build/stability_tests` 并把摘要写入 `results/stage8_stability/coverage.log`，
再以 `compute-sanitizer --tool memcheck --error-exitcode 99` 运行同一矩阵；两段
状态均为 0，错误汇总为 0。大日志只保留在远端 `/data`，仓库保存小型
`coverage.csv` 和校验和。
