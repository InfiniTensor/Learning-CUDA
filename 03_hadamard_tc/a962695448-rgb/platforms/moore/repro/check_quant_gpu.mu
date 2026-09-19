#include <musa_runtime.h>
#define HADAMARD_QUANT_HD __host__ __device__
#include "exact_int4.hpp"
#undef HADAMARD_QUANT_HD
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

struct Record { float value, scale; std::int32_t expected; };
static_assert(sizeof(Record) == 12);
#define CHECK(call) do { auto status=(call); if(status!=musaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,musaGetErrorString(status)); return 2; } } while(0)

__global__ void compare_quant(const Record* input, int* legacy, int* candidate, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto item = input[index];
    const float quotient = __fdiv_rn(item.value, item.scale);
    const float lower = floorf(quotient), fraction = quotient - lower;
    int q = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
    legacy[index] = q < -7 ? -7 : (q > 7 ? 7 : q);
    candidate[index] = hadamard::moore::detail::exact_int4_from_bits(
        __float_as_uint(item.value), __float_as_uint(item.scale));
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"Expected fixture path\n"); return 2; }
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    const auto bytes = file.tellg();
    if (!file || bytes <= 0 || bytes % sizeof(Record) || bytes > 120000000) return 2;
    const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(Record);
    std::vector<Record> records(count);
    file.seekg(0); file.read(reinterpret_cast<char*>(records.data()),bytes);
    if (!file) return 2;
    for (const auto& r : records)
        if (!std::isfinite(r.value) || !std::isfinite(r.scale) || r.scale <= 0 || r.expected < -7 || r.expected > 7) return 2;
    CHECK(musaSetDevice(0));
    musaDeviceProp prop{}; CHECK(musaGetDeviceProperties(&prop,0));
    std::printf("DEVICE %s arch=%d.%d reported_warp=%d\n",prop.name,prop.major,prop.minor,prop.warpSize);
    Record* input=nullptr; int *legacy=nullptr,*candidate=nullptr;
    CHECK(musaMalloc(reinterpret_cast<void**>(&input),count*sizeof(Record)));
    CHECK(musaMalloc(reinterpret_cast<void**>(&legacy),count*sizeof(int)));
    CHECK(musaMalloc(reinterpret_cast<void**>(&candidate),count*sizeof(int)));
    CHECK(musaMemcpy(input,records.data(),count*sizeof(Record),musaMemcpyHostToDevice));
    compare_quant<<<static_cast<unsigned>((count+255)/256),256>>>(input,legacy,candidate,count);
    CHECK(musaGetLastError()); CHECK(musaDeviceSynchronize());
    std::vector<int> a(count),b(count);
    CHECK(musaMemcpy(a.data(),legacy,count*sizeof(int),musaMemcpyDeviceToHost));
    CHECK(musaMemcpy(b.data(),candidate,count*sizeof(int),musaMemcpyDeviceToHost));
    std::size_t old_errors=0,new_errors=0,different=0;
    for (std::size_t i=0;i<count;++i) {
        old_errors += a[i]!=records[i].expected;
        if (b[i]!=records[i].expected) {
            if(new_errors<8) std::printf("MISMATCH index=%zu value=%.9g scale=%.9g expected=%d old=%d new=%d\n",i,records[i].value,records[i].scale,records[i].expected,a[i],b[i]);
            ++new_errors;
        }
        different += a[i]!=b[i];
    }
    std::printf("RESULT cases=%zu legacy_errors=%zu candidate_errors=%zu different=%zu\n",count,old_errors,new_errors,different);
    CHECK(musaFree(input)); CHECK(musaFree(legacy)); CHECK(musaFree(candidate));
    return old_errors || new_errors || different ? 1 : 0;
}
