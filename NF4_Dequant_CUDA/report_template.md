# NF4 Dequant CUDA Report Template

## 1. Scope

- Goal:
- Input format:
- Output format:
- Required constraints:

## 2. Environment

- GPU:
- CUDA:
- Compiler:
- Notes:

## 3. Implementation Summary

### 3.1 Kernel design

- Thread mapping:
- Nibble order:
- Scale path:
- Boundary handling:
- Packed store:

### 3.2 Runtime design

- Host memory strategy:
- Device memory strategy:
- Reuse policy:
- Async/stream policy:

## 4. Accuracy Validation

- Reference:
- Dataset:
- Command:
- MAE:
- Threshold:
- Pass/Fail:

## 5. Performance Snapshot

- Command:
- Kernel time:
- Effective bandwidth:
- Speedup vs reference:

## 6. Nsight Systems Results

### 6.1 Steady-state median

- rounds:
- kernel_avg_ns:
- d2h_total_ns:
- h2d_total_ns:
- cudaMemcpy_api_total_ns:
- cudaMalloc_api_total_ns:
- Interpretation:

### 6.2 Cold-start median

- rounds:
- kernel_avg_ns:
- d2h_total_ns:
- h2d_total_ns:
- cudaMemcpy_api_total_ns:
- cudaMalloc_api_total_ns:
- cudaMallocHost_api_total_ns:
- Interpretation:

## 7. Nsight Compute Snapshot

- Metrics:
- Interpretation:
- Caveat:

## 8. Recommended Params

- production:
- steady-profile:
- cold-profile:

## 9. Repro Commands

```bash
# fill commands
```

## 10. Future Work

- [ ] item 1
- [ ] item 2
- [ ] item 3
