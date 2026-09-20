#pragma once
// Ascend C is a separate compiler backend. The shared format functions retain
// their host/device annotations; runtime adapters are only visible to the host.
#if defined(LP_ASCEND_DEVICE)
#include "kernel_operator.h"
#define __host__
#define __device__ __aicore__
using __half = half;
__aicore__ inline float __half2float(half x) { return float(x); }
__aicore__ inline half __float2half_rn(float x) { return half(x); }
#else
#include <acl/acl.h>
#include <cstring>
#include <stdexcept>
#include <string>
#define __host__
#define __device__
using __half = __fp16;
inline float __half2float(__half x) { return float(x); }
inline __half __float2half_rn(float x) { return __half(x); }
namespace lp_ascend {
inline void checked(aclError e) {
  if (e != ACL_SUCCESS)
    throw std::runtime_error(
        "ACL error " + std::to_string(e) + ": " +
        (aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "no runtime detail"));
}
struct Runtime {
  aclrtStream stream{};
  Runtime() {
    checked(aclInit(nullptr));
    checked(aclrtSetDevice(0));
    checked(aclrtCreateStream(&stream));
  }
  ~Runtime() {
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
  }
};
inline Runtime &runtime() {
  static Runtime r;
  return r;
}
inline aclrtStream stream() { return runtime().stream; }
} // namespace lp_ascend
using cudaError_t = aclError;
using cudaEvent_t = aclrtEvent;
constexpr auto cudaSuccess = ACL_SUCCESS;
constexpr auto cudaMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;
constexpr auto cudaMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;
inline const char *cudaGetErrorString(aclError) {
  auto s = aclGetRecentErrMsg();
  return s ? s : "ACL runtime error";
}
inline aclError cudaMalloc(void **p, size_t n) {
  lp_ascend::runtime();
  return aclrtMalloc(p, n, ACL_MEM_MALLOC_HUGE_FIRST);
}
template <class T> inline aclError cudaMalloc(T **p, size_t n) {
  return cudaMalloc(reinterpret_cast<void **>(p), n);
}
inline aclError cudaFree(void *p) { return aclrtFree(p); }
inline aclError cudaMemcpy(void *d, const void *s, size_t n, aclrtMemcpyKind k) {
  auto e = aclrtSynchronizeStream(lp_ascend::stream());
  return e ? e : aclrtMemcpy(d, n, s, n, k);
}
// Existing end-to-end timing uses pageable vectors. ACL's synchronous copy is
// required for those pointers; kernel-only measurements still use device events.
inline aclError cudaMemcpyAsync(void *d, const void *s, size_t n, aclrtMemcpyKind k) {
  auto e = aclrtSynchronizeStream(lp_ascend::stream());
  return e ? e : aclrtMemcpy(d, n, s, n, k);
}
inline aclError cudaMemset(void *p, int v, size_t n) { return aclrtMemset(p, n, v, n); }
inline aclError cudaGetLastError() { return ACL_SUCCESS; }
inline aclError cudaDeviceSynchronize() {
  return aclrtSynchronizeStream(lp_ascend::stream());
}
inline aclError cudaEventCreate(cudaEvent_t *e) { return aclrtCreateEvent(e); }
inline aclError cudaEventDestroy(cudaEvent_t e) { return aclrtDestroyEvent(e); }
inline aclError cudaEventRecord(cudaEvent_t e) {
  return aclrtRecordEvent(e, lp_ascend::stream());
}
inline aclError cudaEventSynchronize(cudaEvent_t e) { return aclrtSynchronizeEvent(e); }
inline aclError cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
  return aclrtEventElapsedTime(ms, a, b);
}
struct cudaDeviceProp {
  char name[128];
};
inline aclError cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  lp_ascend::runtime();
  const char *s = aclrtGetSocName();
  std::strncpy(p->name, s ? s : "Ascend", 127);
  p->name[127] = 0;
  return ACL_SUCCESS;
}
#endif
