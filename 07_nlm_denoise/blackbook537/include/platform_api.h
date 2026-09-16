#ifndef NLM_PLATFORM_API_H
#define NLM_PLATFORM_API_H

// ============================================================================
// 平台抽象层（PAL, Platform Abstraction Layer）
//
// 统一封装设备运行时 API，算法核心代码 100% 共享，平台差异全部收敛到本文件。
// 编译期平台宏由 Makefile 注入：
//   -DPLATFORM_NVIDIA / -DPLATFORM_METAX / -DPLATFORM_MOORE / -DPLATFORM_ILUVATAR
//
// 说明：NVIDIA、沐曦 MACA、天数 CoreX 均提供 CUDA 兼容运行时（API 同名），
//       摩尔线程 MUSA 使用 musa 前缀，单独适配。
// ============================================================================

#define NLM_STR_IMPL_(x) #x
#define NLM_STR(x) NLM_STR_IMPL_(x)

#if defined(PLATFORM_MOORE)
// ---------------- 摩尔线程 MUSA ----------------
#include <musa_runtime.h>
#include <musa_runtime_api.h>
typedef musaError_t gpuError_t;
typedef musaEvent_t gpuEvent_t;
#define GPU_SUCCESS                musaSuccess
#define GPU_MEMCPY_H2D             musaMemcpyHostToDevice
#define GPU_MEMCPY_D2H             musaMemcpyDeviceToHost
#define GPU_MALLOC(p, b)           musaMalloc((p), (b))
#define GPU_FREE(p)                musaFree(p)
#define GPU_MEMCPY(d, s, b, k)     musaMemcpy((d), (s), (b), (k))
#define GPU_EVENT_CREATE(e)        musaEventCreate(e)
#define GPU_EVENT_DESTROY(e)       musaEventDestroy(e)
#define GPU_EVENT_RECORD(e)        musaEventRecord(e)
#define GPU_EVENT_SYNC(e)          musaEventSynchronize(e)
#define GPU_EVENT_ELAPSED(ms,a,b)  musaEventElapsedTime((ms), (a), (b))
#define GPU_GET_LAST_ERROR()       musaGetLastError()
#define GPU_GET_ERROR_STRING(e)    musaGetErrorString(e)
#define GPU_GET_DEVICE(d)          musaGetDevice(d)
#define GPU_FUNC_SET_MAX_SMEM(f,b) musaFuncSetAttribute((const void*)(f), musaFuncAttributeMaxDynamicSharedMemorySize, (int)(b))
#define GPU_GET_SMEM_OPTIN(d,v)    musaDeviceGetAttribute((v), musaDevAttrMaxSharedMemoryPerBlockOptin, (d))

#else
// ---------------- NVIDIA CUDA / 沐曦 MACA / 天数 CoreX ----------------
#include <cuda_runtime.h>
typedef cudaError_t gpuError_t;
typedef cudaEvent_t gpuEvent_t;
#define GPU_SUCCESS                cudaSuccess
#define GPU_MEMCPY_H2D             cudaMemcpyHostToDevice
#define GPU_MEMCPY_D2H             cudaMemcpyDeviceToHost
#define GPU_MALLOC(p, b)           cudaMalloc((p), (b))
#define GPU_FREE(p)                cudaFree(p)
#define GPU_MEMCPY(d, s, b, k)     cudaMemcpy((d), (s), (b), (k))
#define GPU_EVENT_CREATE(e)        cudaEventCreate(e)
#define GPU_EVENT_DESTROY(e)       cudaEventDestroy(e)
#define GPU_EVENT_RECORD(e)        cudaEventRecord(e)
#define GPU_EVENT_SYNC(e)          cudaEventSynchronize(e)
#define GPU_EVENT_ELAPSED(ms,a,b)  cudaEventElapsedTime((ms), (a), (b))
#define GPU_GET_LAST_ERROR()       cudaGetLastError()
#define GPU_GET_ERROR_STRING(e)    cudaGetErrorString(e)
#define GPU_GET_DEVICE(d)          cudaGetDevice(d)
#define GPU_FUNC_SET_MAX_SMEM(f,b) cudaFuncSetAttribute((const void*)(f), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(b))
#define GPU_GET_SMEM_OPTIN(d,v)    cudaDeviceGetAttribute((v), cudaDevAttrMaxSharedMemoryPerBlockOptin, (d))
#endif

// 统一错误检查宏：任一 GPU API 失败即将错误信息写入 err 并返回 false
#define GPU_CHECK(call, err)                                                          \
    do {                                                                              \
        gpuError_t e_ = (call);                                                       \
        if (e_ != GPU_SUCCESS) {                                                      \
            if (err) *err = std::string("[GPU] ") + GPU_GET_ERROR_STRING(e_) +        \
                            " @ " __FILE__ ":" NLM_STR(__LINE__);                     \
            return false;                                                             \
        }                                                                             \
    } while (0)

#endif // NLM_PLATFORM_API_H
