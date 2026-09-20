#pragma once

// Keep one numeric/kernel implementation. Only runtime names differ for MUSA;
// device intrinsics and matrix fragments are handled explicitly by each backend.
#if defined(LP_ASCEND)
#include "../ascend/platform.hpp"
#elif defined(__MUSACC__)
#include <musa_fp16.h>
#include <musa_runtime.h>
#define cudaDeviceProp musaDeviceProp
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaError_t musaError_t
#define cudaEventCreate musaEventCreate
#define cudaEventDestroy musaEventDestroy
#define cudaEventElapsedTime musaEventElapsedTime
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEvent_t musaEvent_t
#define cudaFree musaFree
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaMalloc musaMalloc
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaMemset musaMemset
#define cudaMemsetAsync musaMemsetAsync
#define cudaSuccess musaSuccess
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
