# Kernel versions

- `fused_v0`: scalar input, all FWHT stages in shared memory, full shared-memory amax reduction, scalar INT8 store.
- `fused_v1`: vectorized input and warp-shuffle amax reduction; FWHT remains entirely in shared memory.
- `fused_v2`: vectorized input, the first five FWHT stages in registers with warp shuffle, remaining stages in shared memory, warp reduction, and packed `char4` output.
- `separate`: vectorized shared-memory Hadamard writing an FP32 intermediate, followed by independent amax and quantize kernels.

Published versions are immutable. Create `fused_v3.cu`, add a new enum and API dispatch entry, then register it in the benchmark to add another optimization without overwriting historical results.
