#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cstdio>
__global__ void k_write(int* o) {
    const __nv_bfloat16 b = __float2bfloat16_rn(1.0f);
    const unsigned short u = __bfloat16_as_ushort(b);
    o[0] = (__ushort_as_bfloat16(u) == b) ? 42 : 7;
}
int main(void) {
    int* d = 0;
    int h = 0;
    if (musaMalloc((void**)&d, sizeof(int)) != musaSuccess) { std::printf("alloc-fail"); return 2; }
    k_write<<<1, 1>>>(d);
    if (musaGetLastError() != musaSuccess) { std::printf("launch-fail"); return 3; }
    musaMemcpy(&h, d, sizeof(int), musaMemcpyDeviceToHost);
    musaFree(d);
    std::printf("%d", h);
    return h == 42 ? 0 : 4;
}
