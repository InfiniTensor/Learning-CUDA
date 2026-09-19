# Moore Threads MTT S4000 / MUSA

Native MUSA adaptation of the Hadamard transform and INT4 fusion. The source
contract and the independent FP64 dense reference are inherited from the
project at `cb82ea6e1d5c8f78b48b6922b0f7af279696cc44`; performance and correctness
must be measured independently on this device.

## Verified delivery status

The exact INT4 optimization passed native S4000 validation on 2026-09-16.
Default Optimized fusion is 2.012–2.898 times as fast as its previous
implementation across all 30 tested shape/dtype configurations and all three
paired rounds. Optional Shuffle32 fusion is 2.884–7.781 times as fast as its own
previous implementation. These are old/new comparisons from the same executable,
not comparisons with a different platform or an application-level speedup.

The actual integer implementation passed 17,527,208 CPU comparisons. A
2,074,568-record overlapping GPU fixture exactly matched both CPU labels and
the original native division path. Full validation passed 1,504 inputs, 192 API
checks and 14 CLI rejection checks. INT4 bytes and scales remain exact.

Three paired processes produced 6,750 original event measurements. Default
Optimized split/fused paths reduce time by at least 5% in every measured round
and configuration. Transform-only times have no consistent improvement.
Shuffle32 still loses to Optimized on some small shapes, so the default remains
Optimized. See [the Chinese report](REPORT.zh-CN.md),
[algorithm derivation](QUANTIZATION_METHOD.md), and
[verification scope and hashes](VERIFICATION.json).

All 77 optimization text files and the byte-identical fixture are backed up
locally. Six Moore source files match the GPU run byte-for-byte. The shared
reference in GitHub differs from the tested copy only by one trailing blank
line; VERIFICATION.json records both hashes.
The previous port's 145 text files are preserved separately. Seven new build
binaries and ten previous executables were not downloaded; the stopped server
has a finite retention period. Source, logs, raw CSVs and fixture are included
in the local optimized delivery ZIP. The core recorded measurements, frozen
control, fixture generator and offline analysis are now available in the
[public reproduction directory](repro/README.zh-CN.md).

The initial device probe compiled and executed on MTT S4000, MUSA SDK/mcc 5.1.0,
driver 5.1.0-server, target `mp_22`. Host device properties report warp size 128;
device-code `warpSize` is 32. The diagnostic observed zero errors for width-32
shuffle, 256 errors for width-64, and 512 for width-128. All 512 simple FP16/BF16
conversion checks passed. Those wider shuffle experiments are retained as
negative evidence and are not enabled in the implementation.

The first quick run found a one-ULP row-scale difference for FP16 input
`0.5009765625 / 7`: ordinary device division produced bits `1033015882`, while
the CPU reference produced `1033015881`. An additional compiler divide/sqrt
flag did not change that case. Explicit native `__fdiv_rn` closed the difference
and passed the quick matrix without changing reference tolerances. A permanent
scale witness now exercises all three methods and both dtypes.

## API and implementation

- `hadamard_api.h` declares asynchronous `transform`, `quantize_int4`, and
  `transform_int4`, using native `musaStream_t`, `__half`, and `__mt_bfloat16`.
- Inputs are contiguous `[rows, dim]`; a four-dimensional tensor is flattened
  as `rows = batch * seq * heads`. `dim` supports powers of two from 1 to 256.
- Storage is FP16/BF16; arithmetic is FP32. Fused quantization first rounds the
  transform to its public storage type. INT4 is symmetric `[-7,7]`, nearest-even,
  with even elements in low nibbles and one FP32 scale per row.
- `Baseline` uses one thread per element with two barriers per butterfly stage.
- `Optimized`, the default, gives each thread exclusive ownership of a butterfly
  pair and uses one barrier per stage. It makes no warp-width assumption.
- Optional `Shuffle32` uses 32-lane logical row groups and local-register higher
  butterfly stages. All 256 threads execute each exchange even when the last
  row batch is incomplete. It is gated by a dedicated device probe.
- Optimized and Shuffle32 quantization use exact integer comparisons that
  preserve both binary32 quotient rounding and ties-to-even integer rounding.
  Row scales retain native correctly rounded division; Baseline retains the
  original per-element division path. See exact_int4.hpp.
- The API does not allocate, copy, or synchronize. Exact in-place transformation
  is supported; partial overlap and quantization buffer overlap are rejected.
  The caller must supply valid, adequately sized buffers on the current device.
- FP8, autograd, arbitrary strides, and dimensions above 256 are outside this port.

## Reproduce

Run from this project's `03_hadamard_tc/a962695448-rgb` directory on Linux with
the installed MUSA SDK. The runner changes environment variables only for its
subprocesses; it does not replace the driver or framework.

```bash
python3 platforms/moore/run_platform.py --probe-only --output results/moore/probe-new
python3 platforms/moore/run_platform.py --quick --shuffle32 --no-benchmark --output results/moore/quick-new
python3 platforms/moore/run_platform.py --shuffle32 --output results/moore/full-new
```

Each output directory must be new. Without `--shuffle32`, the runner tests the
baseline/shared paths and verifies that unavailable Shuffle32 calls fail clearly.
The default full run performs three separate benchmark processes, five timing
groups per method/configuration, and 100 calls per event interval.

The commands above reproduce validation and the current-version baseline
benchmark. The [public reproduction guide](repro/README.zh-CN.md) contains the
old/new paired harness, frozen control, raw measurements and CPU fixture
generator. Offline reanalysis uses only the Python standard library. The paired
6,750-row dataset and older 4,050-row baseline dataset are separate experiments.

## Validation and timing boundaries

The matrix covers both dtypes, all supported dimensions, multiple row counts,
normalization, seeded uniform/normal/outlier data, zero/impulse inputs, guarded
and minimally aligned buffers, exact in-place behavior, non-default streams,
large grid-stride tails, API rejection cases, and 14 invalid CLI cases. The
oracle computes an independent FP64 dense Hadamard product and rounds to the
public output dtype. FP16/BF16 strict absolute tolerances remain 0.01/0.05.
CPU, separate, and fused INT4 bytes/scales must agree exactly. Hand-written
rounding witnesses include positive/negative ties and `[0.75, 0.25] -> 0x37`.

Every benchmark method is checked before timing and after its final timing
group. Large timing shapes use three independently checked dense-reference rows
plus full-array method equality and INT4 checks; this is stated separately from
the all-element dense oracle used in the validation matrix.

MUSA event timing excludes allocation, host/device transfers and output checks.
It can include device idle gaps caused by host submission; it is not application
latency. CSV provides microseconds and milliseconds. Logical I/O throughput is an
estimate, not measured physical bandwidth. Slower cases remain in the report.

## Sources

- [MUSA 5.1 MCC manual](https://docs.mthreads.com/musa-sdk/version-5.1.0/toolkits/mcc_compiler/)
- [MUSA warp functions](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/programming_guide/musa_cpp_syntax/warp_functions/)
- `SOURCE_PROVENANCE.json` records the original source and local snapshot hashes.

Results apply to the recorded MUSA 5.1.0 / S4000 environment and measured input matrix.
