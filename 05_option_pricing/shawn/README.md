# Task 05 - CUDA Derivatives Pricing and Risk Estimation

This project implements Monte Carlo pricing for European calls and puts,
arithmetic-average Asian calls, and discretely monitored up-and-out barrier
calls. The primary backend is CUDA; the portable CPU implementation is the
correctness reference and lets the project build on machines without CUDA.

The advanced path also supports American calls/puts through
Longstaff-Schwartz, Black-Scholes/Heston/local-volatility dynamics,
randomized-Halton QMC, and common-random-number Delta/Gamma/Vega on CPU and CUDA.
An experimental multi-GPU partitioning interface is present, but has not been
run or validated on multiple GPUs. All reported GPU measurements use one RTX
5090; implementation availability does not establish numerical accuracy for
every model and parameter combination.

## Final report and evidence

Read [REPORT.md](REPORT.md) for the measured results and their limitations,
[the reproducibility record](docs/report_reproducibility.md) for the frozen
configuration and requirement mapping, and
[the evidence manifest](report_assets/evidence/manifest.json) for CSV/log hashes.
The submission package also includes the exported report as `REPORT.pdf`.

## Implemented contract

- Text `key = value` parsers matching the project statement.
- Geometric Brownian motion under the risk-neutral measure.
- European call/put validation against the Black-Scholes analytic price.
- Asian call and barrier call validation against a CPU Monte Carlo reference.
- `none`, `antithetic`, and terminal-asset `control_variate` estimators.
- Black-Scholes, non-negative projected-Euler Heston, and power-law local volatility.
- American call/put Longstaff-Schwartz with quadratic continuation regression.
- Delta, Gamma, and Vega from central finite differences with common random
  numbers; Heston Vega bumps the initial volatility `sqrt(v0)`.
- Randomized Halton with independent shift replications and
  replication-level standard error on CPU and CUDA.
- Experimental multi-GPU path partitioning and estimate merging; not multi-GPU validated.
- Price standard error and an approximate, normal-based two-sided 95% confidence interval.
- CUDA path simulation with cuRAND Philox or a custom SplitMix64/Box-Muller RNG.
- Per-block FP64 moment reduction followed by a separate five-block reduction
  kernel, so simulation and reduction time are measured independently.
- Result and performance logs, including timing scope and paths/second.

The Asian average uses the asset value at the end of each monitoring step. The
barrier option is an up-and-out call; the barrier is checked at those same
discrete times. These choices are explicit because the assignment leaves the
monitoring convention open.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake enables CUDA automatically when `nvcc` is available. To require a
specific architecture on the training server, configure for example with
`-DCMAKE_CUDA_ARCHITECTURES=75` for T4. On a non-CUDA machine the CPU backend
and all CPU tests still build.

## Run

Smoke test:

```bash
./build/derivatives_pricer \
  --option configs/european_call.txt \
  --simulation configs/simulation_smoke.txt \
  --backend auto \
  --output outputs/european_call.txt \
  --performance outputs/european_call.performance.log
```

Path-dependent example:

```bash
./build/derivatives_pricer \
  --option configs/asian_call.txt \
  --simulation configs/simulation_full.txt \
  --backend cuda \
  --reference-paths 200000 \
  --cpu-benchmark-paths 200000 \
  --output outputs/asian_call.txt \
  --performance outputs/asian_call.performance.log
```

For ordinary Monte Carlo, `num_paths` counts estimator samples. In antithetic
mode, each sample evaluates both `Z` and `-Z`, so `physical_paths` is twice
`num_paths`. For randomized Halton, it counts points across shift replications;
uncertainty is estimated across independent shifts, not across individual points.
The performance log separates CUDA-event simulation/reduction intervals and
pricing-call wall time. Event intervals are not pure profiler kernel durations.
It also runs a configurable single-thread
CPU comparison and reports throughput-normalized speedup. For path-dependent
contracts, `--reference-paths` controls an independently seeded numerical
reference; `--cpu-benchmark-paths` controls a separate single-thread CPU run
with the same model, steps, RNG path type, and variance-reduction mode as the
CUDA run. If the second option is omitted it inherits `--reference-paths` for
compatibility with older commands. Set either value to `0` when that evidence
is not required. Reference generation and result-file I/O are outside the
pricing-call timing boundary.

## Advanced CPU/GPU runs

Old configuration files remain valid. Advanced option files add `model`;
advanced simulation files add `qmc_replications`, `num_gpus`,
`compute_greeks`, and finite-difference bump sizes.

```bash
# American put with LSM and Greeks
./build/derivatives_pricer \
  --option configs/american_put.txt \
  --simulation configs/simulation_advanced_smoke.txt \
  --backend cpu --reference-paths 0 \
  --output outputs/american_cpu.txt \
  --performance outputs/american_cpu.log

# Heston randomized-Halton run on one GPU
./build/derivatives_pricer \
  --option configs/heston_call.txt \
  --simulation configs/simulation_qmc_smoke.txt \
  --backend cuda --reference-paths 0 \
  --output outputs/heston_qmc_gpu.txt \
  --performance outputs/heston_qmc_gpu.log
```

American CUDA LSM generates paths, computes block regression moments, and applies
exercise decisions on the GPU. The host aggregates the block moments, solves
the small 3x3 continuation-regression system, and computes final statistics from
the returned cash flows. These transfers and host work are included in
pricing-call wall time.

Before using GPU results in the report, run the automated accuracy matrix:

```bash
python3 scripts/run_gpu_validation.py \
  --binary build/derivatives_pricer \
  --output-dir outputs/gpu_validation

# Optional future validation on a server exposing at least two GPUs:
# This is not part of the measurements reported in REPORT.md.
python3 scripts/run_gpu_validation.py \
  --binary build/derivatives_pricer \
  --output-dir outputs/gpu_validation_multi \
  --include-multi-gpu
```

The matrix covers GBM/Heston/local volatility, pseudo-random and Halton paths,
Greeks, and American LSM. Prices use a joint-standard-error gate; Greeks use
explicit absolute/relative tolerances. American pseudo-random validation uses
the dedicated `simulation_american_validation.txt` configuration with 200,000
paths and a 1% spot bump because finite-difference LSM Gamma is especially
sensitive to path noise. American Halton uses the dedicated
`simulation_american_qmc_validation.txt` configuration with 131,072 paths and
64 steps; other cases retain the faster smoke configurations.

The driver runs every requested case even if an earlier gate fails, writes
`summary.csv`, `environment.txt`, and a report-ready `report_table.md`, then
exits non-zero if any case failed. Retain the per-case CPU/GPU result and
performance logs beside those three summary artifacts. Exact CPU/GPU Halton
agreement is implementation-parity evidence; American Greek accuracy must also
be checked against the independent CRR tree reference. The generated comparison
rows therefore contain separate backend-parity and reference-accuracy fields.
Black-Scholes European rows use analytic price/Greeks; Black-Scholes American
rows use a 2,000-step CRR price and finite-difference CRR Greeks. Models without
an independent reference are explicitly marked `n/a` rather than silently
treated as correct.

## Reproducible experiment sweep

Use repeated, independently seeded replications for convergence and
variance-reduction claims:

```bash
python3 scripts/run_sweep.py \
  --binary build/derivatives_pricer \
  --option configs/asian_call.txt \
  --backend cpu \
  --paths 10000 100000 1000000 \
  --variance none antithetic control_variate \
  --steps 252 \
  --master-seed 20260824 \
  --replications 20 \
  --reference-paths 1000000 \
  --cpu-benchmark-paths 0 \
  --output-dir outputs/asian_sweep
```

Each replication derives one seed from `master_seed`; that seed is reused
across path counts and variance-reduction modes. This common-random-number
pairing makes method comparisons less noisy, while different replications
remain independent. `summary.csv` contains every raw run and `aggregate.csv`
contains empirical price SD, mean reported SE, SD/SE calibration ratio,
relative SE, `SE^2 * runtime`, error, runtime, throughput, and optional CPU
speedup. Kernel/reduction time and the CPU benchmark's paths, milliseconds, and
throughput remain separate columns. `report_table.md` is the report-ready
rendering. Preserve all three
files and report the master seed, replication count, effective RNG
implementation (`rng`), requested CUDA RNG (`rng_requested`), and seed
derivation rule.

For Asian and barrier options, the sweep computes one fixed, independently
seeded CPU control-variate reference and reuses it for every tested setting.
Its uncertainty and seed are recorded beside the raw estimates. A standalone
path-dependent run uses `--reference-paths N` in the same way; set it to `0`
only for a quick run where no absolute-error claim is needed.

## Final report experiment suite

First check the complete pipeline with the inexpensive preset. Its results are
diagnostic only and must not be copied into the final report:

```bash
python3 scripts/run_task5_experiments.py \
  --binary build-cuda128/derivatives_pricer \
  --preset quick \
  --backend cuda \
  --include-barrier \
  --output-dir outputs/task5_quick_run
```

Then run the immutable report preset from a clean Git commit:

```bash
python3 scripts/run_task5_experiments.py \
  --binary build-cuda128/derivatives_pricer \
  --preset report \
  --backend cuda \
  --include-barrier \
  --profile-tools nsys \
  --output-dir outputs/task5_report_run
```

The report preset executes the latest CPU/GPU reference gates, repeated MC
sweeps over `10^4` through `10^7` paths, a randomized-Halton sweep, and seven
10M-path/256-step Asian performance replications. Profiles use the separate
medium-size `simulation_profile.txt`; they do not replace the 10M performance
measurements. The root output directory contains the environment, exact plan,
stage manifest, console logs, an artifact index, and merged Markdown tables.
Dependency-free SVG convergence/error curves are generated from the aggregate
CSVs under `plots/`; they are never computed from rounded Markdown values.
The driver refuses a dirty worktree or a non-empty evidence directory so stale
files cannot silently enter the report. The task-5 driver does not implement
`--resume`: preserve interrupted logs and choose a fresh output directory for a
new run. Add `ncu` only when GPU performance counters are accessible; the archived
experiments have usable nsys traces but no usable NCU counter results. Omit
`--profile-tools` if Nsight is not installed, then collect profiles separately
when the tools become available.

## Profiling checklist

Run the release build on the target GPU before making performance claims:

```bash
ncu --set full --target-processes all ./build/derivatives_pricer \
  --option configs/asian_call.txt --simulation configs/simulation_profile.txt \
  --backend cuda --output outputs/asian.txt --performance outputs/asian.log

nsys profile --stats=true -o outputs/asian_nsys ./build/derivatives_pricer \
  --option configs/asian_call.txt --simulation configs/simulation_profile.txt \
  --backend cuda --output outputs/asian.txt --performance outputs/asian.log
```

`scripts/profile_cuda.py` collects the same ncu/nsys artifacts and records the
exact command. Profile before changing block sizes, path layout, LSM regression
reductions, or kernel fusion; no profiling-driven speedup is claimed until raw
profiles exist on the target GPU.

Record the GPU model, CUDA version, compiler flags, option configuration,
number of paths, monitoring steps, variance-reduction mode, kernel time,
end-to-end time, confidence interval, and CPU comparison in `REPORT.md`.
