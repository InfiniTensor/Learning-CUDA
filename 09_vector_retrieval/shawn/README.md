# Task 09 - GPU Vector Search Engine

This project provides a correctness-first GPU exact-search baseline and a
persistent IVF-Flat approximate index. It supports L2 distance, inner product,
and cosine similarity; FP32 and FP16 input files; batched query files; and
configurable Top-K up to 100.

## Final report and evidence

Read [REPORT.md](REPORT.md) for the single-RTX-5090 measurements,
[the reproducibility record](docs/report_reproducibility.md) for experiment
parameters and requirement mapping, and
[the evidence manifest](report_assets/evidence/manifest.json) for CSV/log hashes.
The submission package also includes the exported report as `REPORT.pdf`.
Formal performance uses 1M vectors of dimension 128 and 1,000 queries; reported
recall uses only the first 20 queries on the clustered synthetic dataset.

## Implemented contract

- GPU exact distance/similarity calculation.
- GPU Top-K baseline using CUB radix sort, with results ordered by score.
- CPU exact implementation used as the correctness and recall ground truth.
- IVF-Flat training, inverted-list construction, save/load, and `nprobe` search.
- Optional CUDA assignment during k-means training and final index assignment.
- Result file, index file, QPS, P50/P99 latency, memory, and quality logs.
- CPU-exact throughput comparison and a CUDA working-set estimate that includes
  database, candidate, score/ID, and CUB temporary buffers.
- `recall@K` and mean absolute score difference at corresponding ranks.
- K values from 1 through 100; run K=1/10/50/100 experiments from config files.

The exact CUDA baseline processes every query in the input and keeps the
database resident on the GPU for the search call. Queries are grouped by
`batch_size`, but this first baseline executes queries sequentially within each
batch so that the full distance vector can be radix-sorted in bounded memory.
This is a baseline, not a concurrent GPU batch implementation. The measured
exact kernel time is dominated by scoring, whereas IVF kernel time is dominated
by sorting. Coalesced scoring, device-side candidate handling, and replacing
full sorting with Top-K selection are proposed optimizations, not measured gains.

## Binary formats

All integer fields are little-endian and all payloads are row-major.

Database header (28 bytes):

| Field | Type | Value |
|---|---|---|
| magic | 8 bytes | `VECDB01\0` |
| version | uint32 | 1 |
| num_vectors | int64 | N |
| dim | int32 | D |
| dtype | uint8 | 1=FP32, 2=FP16 |
| metric | uint8 | 1=L2, 2=inner product, 3=cosine |
| reserved | uint16 | 0 |

The database payload is `N * D` elements in the declared dtype.

Query header (28 bytes): `VECQRY1\0`, uint32 version, int64 query count,
int32 dimension, uint8 dtype, and three zero reserved bytes. The payload is
`num_queries * D` elements.

The IVF index stores `IVFFLT1\0`, version, N, D, `nlist`, metric, FP32 centers,
uint64 list offsets, and int64 vector IDs. Original vectors remain in the
database file, so loading an index never retrains it but does not duplicate the
vector payload.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is enabled automatically when `nvcc` is present. Set the training-server
architecture explicitly when known, for example
`-DCMAKE_CUDA_ARCHITECTURES=75` for T4. A CPU-only build remains available for
format, index, and correctness tests.

## Generate smoke data

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/generate_data.py \
  --output-dir data/smoke --num-vectors 10000 --num-queries 100 \
  --dim 128 --dtype fp32 --metric l2 --clusters 64
```

For the required scale, regenerate with at least `--num-vectors 1000000` and
`--num-queries 1000`. Record data generation parameters and file hashes.

## Exact search

```bash
./build/vector_search search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --config configs/exact_smoke.txt \
  --backend auto \
  --output outputs/exact.results.txt \
  --performance outputs/exact.performance.log \
  --quality outputs/exact.quality.log
```

## IVF-Flat build and search

```bash
./build/vector_search build \
  --database data/smoke/database.bin \
  --config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend auto \
  --performance outputs/ivf.build.log

./build/vector_search search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend auto \
  --output outputs/ivf.results.txt \
  --performance outputs/ivf.performance.log \
  --quality outputs/ivf.quality.log
```

The quality log compares the first `quality_queries` against CPU exact search.
`recall_at_k` measures set overlap and `rank_id_agreement` measures exact ID
agreement at each returned rank. The validation gate requires identical Top-K
ID sets, but floating-point near-ties can reduce strict rank agreement below
1.0. Such swaps pass only with tolerance-aware rank agreement and score
agreement after matching the same IDs. `mean_score_abs_error` in the quality
log is rank-aligned; the validation matrix additionally records ID-aligned errors.
Increase `quality_queries` for the final report after measuring the validation
cost.
Sweep both `nprobe` and `batch_size`; do not present a single operating point as
the whole latency/recall trade-off.

For repeated CPU or GPU measurements, use the sweep driver:

```bash
python3 scripts/run_sweep.py \
  --binary build/vector_search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --base-config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend cpu \
  --top-k 1 10 50 100 \
  --nprobe 1 4 8 16 \
  --batch-size 1 32 128 \
  --repetitions 5 \
  --output-dir outputs/ivf_sweep
```

`summary.csv` preserves each repetition. `aggregate.csv` reports mean and
median QPS, QPS standard deviation, mean P50/P99 latency, recall, rank
agreement, candidate count, CPU-reference QPS, speedup, and memory estimates.
The driver runs `--warmup` unmeasured repetitions for every configuration and
stores every measured result/quality/performance/console log under `raw/`.
It also records exact commands, input/index SHA-256 values, and the GPU/software
environment. Exact mode is supported by omitting `--index` and using an exact
base config.

## CUDA correctness matrix

Before using any GPU number in the report, build with CUDA, run CTest, then
execute the automated CPU/GPU gate:

```bash
ctest --test-dir build --output-on-failure
python3 scripts/run_gpu_validation.py \
  --binary build/vector_search \
  --output-dir outputs/gpu_validation
```

The default matrix covers L2/inner-product/cosine, FP32/FP16,
K=1/10/50/100 exact search, CUDA IVF assignment, and CPU/CUDA reranking of the
same persistent index. Each hard gate requires an identical per-query Top-K ID
set, score agreement after matching the same vector IDs, and tolerance-aware
rank agreement. Strict rank agreement remains in the report as a diagnostic:
CPU/CUDA-near-tie swaps are accepted only when their rank scores are within
`--score-tolerance`. Exactly repeated scores still use the global
smaller-vector-ID tie policy and are covered by a strict CUDA CTest. The driver
attempts the remaining matrix after an individual failure, exits non-zero at
the end, and retains data hashes, exact commands, environment information, raw
result/quality/performance logs, `summary.csv`, and `report_table.md`. Its
default 4096-vector dataset is a correctness matrix, not the required
one-million-vector performance run.

## End-to-end formal experiment

Install both the full-scale NumPy generator and report plotting dependency in
the virtual environment created above (on Debian/Ubuntu, install the distribution's
`python3-venv` package if environment creation is unavailable):

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run a short GPU pipeline first. Its numbers are smoke evidence only:

```bash
python3 scripts/run_task9_experiments.py \
  --binary build/vector_search \
  --output-dir outputs/task9_quick \
  --preset quick
```

The report preset fixes `N=1,000,000`, `D=128`, `Q=1,000`, FP32/L2,
`nlist=4096`, K=1/10/50/100, an `nprobe` sweep, a batch-size sweep, warm-up,
and repeated measurements:

```bash
python3 scripts/run_task9_experiments.py \
  --binary build/vector_search \
  --output-dir outputs/task9_report \
  --preset report
```

The top-level directory contains `environment.txt`, `experiment_plan.json`,
`manifest.json`, `artifact_index.md`, and `report_tables.md`. Data metadata and
hashes, the persistent index, every raw repetition, aggregate CSVs, and SVG/PNG
plots remain in their stage directories. A failed long run can continue only
with the identical command plus `--resume`; the driver verifies the frozen plan
and expected artifacts before skipping a completed stage.

The GPU exact and IVF throughput/latency rows use all 1,000 queries. Each search
also computes CPU exact ground truth for the explicitly logged
`quality_queries` subset (20 in the report preset), so the CPU-reference QPS and
speedup remain practical and their smaller sample size is never confused with
the full GPU workload.

## Profiling and next optimization

Profile exact and IVF modes separately. The initial exact path materializes N
scores and IDs and performs a full radix sort per query. `ncu` focuses on the
project score kernel; `nsys` retains the complete timeline, including CUB,
transfers, synchronization, and IVF host coarse selection:

```bash
python3 scripts/profile_vector_search.py --tool ncu \
  --binary build/vector_search \
  --database outputs/task9_report/data/database.bin \
  --queries outputs/task9_report/data/queries.bin \
  --config outputs/task9_report/configs/exact_profile.txt \
  --output-dir outputs/task9_report/profile_ncu_exact

python3 scripts/profile_vector_search.py --tool nsys \
  --binary build/vector_search \
  --database outputs/task9_report/data/database.bin \
  --queries outputs/task9_report/data/queries.bin \
  --config outputs/task9_report/configs/ivf_profile.txt \
  --index outputs/task9_report/index/database.ivf \
  --output-dir outputs/task9_report/profile_nsys_ivf
```

Alternatively, add `--profile-tools ncu nsys` to the original formal command
when both tools are available. Keep that exact option when using `--resume`.

`batch_size` currently groups queries at the API/scheduler level, while each
query still launches and sorts sequentially inside the CUDA backend. The batch
sweep is therefore an honest diagnostic: a flat curve is expected evidence of
the current limitation, not evidence of concurrent batching.

The current report is tied to the archived single-GPU runs, not to hypothetical
results from these example commands. Valid nsys traces are archived; NCU counters
were unavailable because of host permission restrictions. For a new experiment,
retain GPU/CUDA/compiler information, input hashes, N/D/Q, dtype, metric, K,
batch size, nlist/nprobe, QPS, P50/P99, memory scope, and recall sample size.
