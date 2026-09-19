# Exact INT4 comparison optimization

The target contract is `clamp(RNE_integer(RNE_float32(value / scale)), -7, 7)`.
The public scale remains the original correctly rounded FP32 `max(abs(x))/7`.
Only per-element quantization in Optimized and Shuffle32 changes; Baseline
retains the original native division implementation.

## Why ordinary thresholds are insufficient

The first rounding step maps an interval of exact ratios to each binary32
half-integer. The following integer tie must go to the even integer. Therefore,
comparing only against `(k + 0.5) * scale` would not preserve the contract.

For the transition between positive integers `k` and `k+1`:

- If `k` is even, transition only above `k+0.5 + half_ulp(k+0.5)`.
- If `k` is odd, transition at or above `k+0.5 - half_ulp(k+0.5)`.

The half-integers have even binary32 significands. At k=0, the relevant upper
half-ULP is 2^-25. For k=1, k=2..3, and k=4..6 the half-ULP is respectively
2^-24, 2^-23, and 2^-22. Expressing all boundaries over 2^25 gives integer
numerators no larger than 218103816. Multiplication by a 24-bit input
significand fits in 52 bits. The implementation uses a three-step binary search
over these seven increasing boundaries and restores the sign at the end.

The exponent handling also represents binary32 subnormals. A negative alignment
shift means the value is below every transition. A shift of at least 30 means
it is above every transition; these cases avoid unsafe shifts. The remaining
left shifts occupy at most 53 bits. This proof assumes finite value and finite
positive scale in the row-quantization contract; it does not define NaN handling.

## Validation boundaries

The actual portable C++ implementation is tested against the existing project
CPU divide/round helper, not against another copy of the integer algorithm.
Checks include all finite nonnegative FP16/BF16 encodings under selected row
maxima, sign reversals, signed zero, subnormals and binary32 rounding-boundary
neighbors. There are 17,527,208 CPU checks; they are not a claim of exhaustive
coverage of every possible pair of binary32 inputs.

The 2,074,568-record GPU fixture is a subset of these classes and must be checked
against both the original native `__fdiv_rn` path and independently generated
CPU labels. GPU and CPU counts overlap and must not be added together.

The paired benchmark links the frozen control in a separate namespace and the
candidate in one executable. Their inputs, output checks, stream and timing
protocol are shared. Three independent benchmark processes rotate method order.
Both full Hadamard regression and the disabled-Shuffle32 compatibility build
must pass before the candidate is accepted.

The acceptance target is the default Optimized split/fused INT4 paths over the
existing 30 shape/dtype configurations. Each must preserve exact results and
reduce event time by at least 5% in all three paired rounds. This criterion is
recorded before inspecting the paired results. Transform-only and optional
Shuffle32 results are reported in full, including regressions.
