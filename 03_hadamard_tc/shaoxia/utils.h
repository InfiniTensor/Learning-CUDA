#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32

#define ATOMIC_M 16
#define ATOMIC_N 8
#define ATOMIC_K 16




__device__ __forceinline__
void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr)
        : "memory"
    );
}


__device__ __forceinline__
void cp_async_commit() {
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory"
    );
}


template<int N>
__device__ __forceinline__
void cp_async_wait() {
    static_assert(N >= 0 && N <= 7,
                  "cp.async.wait_group immediate must be in valid range");

    asm volatile(
        "cp.async.wait_group %0;\n"
        :
        : "n"(N)
        : "memory"
    );
}


// Most common: wait for every previously committed async copy.
__device__ __forceinline__
void cp_async_wait_all() {
    asm volatile(
        "cp.async.wait_all;\n"
        :
        :
        : "memory"
    );
}