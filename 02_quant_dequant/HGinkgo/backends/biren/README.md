# Biren Backend

Load BIRENSUPA, then configure the SUPA build for the installed architecture:

```bash
source /usr/local/birensupa/br_container_tools/brsw_set_env.sh
cmake -S ../.. -B build-biren -G Ninja -DLP_BACKEND=biren -DLP_BIREN_ARCH=br100
cmake --build build-biren
```

The `.su` adapters compile the shared CUDA-style quantization source through
BIRENSUPA. `include/cuda_runtime.h` maps the project runtime calls to SUPA,
keeping the packed MXFP8/NVFP4 implementation single-sourced.
