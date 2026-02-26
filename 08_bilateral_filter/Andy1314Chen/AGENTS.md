# AGENTS.md - Bilateral Filter CUDA Project

CUDA bilateral filter implementation for InfiniTensor training camp.

## Build Commands

```bash
make                # Build all (requires CUDA toolkit + OpenCV4)
make clean          # Clean build artifacts
```

**Dependencies**: CUDA toolkit (>= 11.0), OpenCV4 (`pkg-config --exists opencv4`), C++17, GNU Make

**Compiler flags** (see `Makefile`):
```makefile
CXXFLAGS  = -std=c++17 -O3 -Wall -Wextra -I./include -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -arch=sm_89 -I./include   # adjust -arch to target GPU
```

## Run Commands

### Single Implementation
```bash
./bilateral_filter input.raw params.txt output.raw              # CPU only
./bilateral_filter --cuda input.raw params.txt output.raw       # CUDA only
./bilateral_filter --opencv input.raw params.txt output.raw     # OpenCV only
```

### Benchmarking (Recommended)
```bash
# CUDA vs OpenCV (skip slow CPU baseline)
./bilateral_filter --bench tests/test_data/input_4k.raw tests/test_data/params.txt

# Full comparison: CPU vs CUDA vs OpenCV
./bilateral_filter --compare-all tests/test_data/input_1080p.raw tests/test_data/params.txt
```

### Switch CUDA Implementation Mode
```bash
BILATERAL_MODE=0 ./bilateral_filter --bench ...  # STANDARD: runtime radius, MAE 0.65
BILATERAL_MODE=1 ./bilateral_filter --bench ...  # TEMPLATE: compile-time radius (default), MAE 0.80
BILATERAL_MODE=2 ./bilateral_filter --bench ...  # SEPARABLE: H+V passes, fastest + lowest MAE 0.15/0.45
BILATERAL_MODE=4 ./bilateral_filter --bench ...  # ADAPTIVE: per-pixel Sobel gradient → radius, edge-preserving
```

## Test Data

Located in `tests/test_data/`:
- `input_1080p.raw`, `input_4k.raw` - RGB images (1920x1080, 3840x2160)
- `input_1080p_gray.raw`, `input_4k_gray.raw` - Grayscale images
- `params.txt` - Default params: radius=5, sigma_spatial=3, sigma_color=30

### Generate Test Image
```bash
python3 -c "
import struct
w, h, c = 1920, 1080, 3
with open('input.raw', 'wb') as f:
    f.write(struct.pack('iii', w, h, c))
    for y in range(h):
        for x in range(w):
            f.write(struct.pack('BBB', (x*3)%256, (y*5)%256, ((x+y)*2)%256))
"
```

## Profiling

```bash
ncu --set full -o profile ./bilateral_filter --cuda input.raw params.txt output.raw
nsys profile -o timeline ./bilateral_filter --cuda input.raw params.txt output.raw
```

## Code Style

### File Organization
```
08_bilateral_filter/
├── src/                            # Source files
│   ├── main.cpp                    # Entry point
│   ├── bilateral_filter_cpu.cpp    # CPU implementation
│   ├── bilateral_filter_cuda.cu    # CUDA implementation (5 modes)
│   ├── bilateral_filter_opencv.cpp # OpenCV wrapper
│   └── image_io.cpp                # Raw image I/O
├── include/                        # Headers
├── tests/test_data/                # Test images and params
├── Makefile
├── PROFILER.md                     # Performance results
└── REPORT.md                       # Optimization notes
```

### Naming Conventions
| Element          | Convention        | Example                      |
|------------------|-------------------|------------------------------|
| Files            | snake_case        | `bilateral_filter_cuda.cu`   |
| Classes/Structs  | PascalCase        | `ImageData`, `FilterParams`  |
| Functions        | snake_case        | `apply_bilateral_filter_cpu` |
| CUDA kernels     | `k_` prefix       | `k_bilateral_filter`         |
| Variables        | snake_case        | `sigma_spatial`              |
| Constants/Macros | UPPER_SNAKE_CASE  | `MAX_RADIUS`, `CUDA_CHECK`   |

### Include Order
```cpp
// 1. Standard library
#include <cstdio>
#include <vector>

// 2. CUDA headers
#include <cuda_runtime.h>

// 3. Third-party
#include <opencv2/opencv.hpp>

// 4. Project headers
#include "image_io.h"
#include "bilateral_filter.h"
```

### Formatting
- 4-space indentation
- Header guards: `#ifndef FILE_NAME_H_` / `#define FILE_NAME_H_` / `#endif  // FILE_NAME_H_`
- Comments in English
- Doxygen for public APIs:
```cpp
/**
 * @brief Apply bilateral filter
 * @param input Input image data
 * @param output Output buffer
 * @param params Filter parameters
 */
```

### Types
- Image pixels: `float` (processing), `uint8_t` (storage)
- Dimensions: `int` or `size_t`
- GPU pointers: use `__restrict__` when applicable

### CUDA Conventions
```cpp
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel launch pattern
dim3 block(BLOCK_X, BLOCK_Y);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
k_bilateral_filter<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
```

### Error Handling
- Return `bool` for I/O functions
- Print to stderr with context: `fprintf(stderr, "Failed to open: %s\n", path);`
- Use `CUDA_CHECK()` macro for all CUDA calls

## Validation

- Correctness: MAE < 1.0 vs OpenCV `bilateralFilter`
- Performance metrics: time (ms), throughput (MP/s), speedup ratio
- Record results in `PROFILER.md`
- Document optimizations in `REPORT.md`

## Quick Reference

| Task                  | Command                                                    |
|-----------------------|------------------------------------------------------------|
| Build                 | `make`                                                     |
| Run benchmark         | `./bilateral_filter --bench tests/test_data/input_4k.raw tests/test_data/params.txt` |
| Test grayscale        | `./bilateral_filter --bench tests/test_data/input_4k_gray.raw tests/test_data/params.txt` |
| Profile with ncu      | `ncu --set full -o prof ./bilateral_filter --cuda ...`     |
| Use separable mode    | `BILATERAL_MODE=2 ./bilateral_filter --bench ...`          |


## 参考资料
- https://github.com/xytroot/Bilateral-Filter/tree/main
- https://docs.opencv.org/4.10.0/d0/d05/group__cudaimgproc.html
- https://docs.nvidia.com/vpi/algo_bilat_filter.html
- https://github.com/CVCUDA/CV-CUDA/blob/main/DEVELOPER_GUIDE.md
- https://github.com/JuliaParallel/rodinia/blob/master/cuda/_bilateral/bilateralFilter.cpp
- https://github.com/WolframRhodium/VapourSynth-BilateralGPU