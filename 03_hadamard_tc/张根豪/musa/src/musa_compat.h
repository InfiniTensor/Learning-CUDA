#pragma once

#include <musa_runtime.h>
#include <musa_bf16.h>

// The original benchmark uses CUDA names. MUSA exposes the same runtime
// operations with musa* names, so keep the copied kernel source unchanged.
#define cudaStream_t musaStream_t
#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaDeviceAttr musaDeviceAttr
#define cudaDeviceProp musaDeviceProp
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaGetLastError musaGetLastError
#define cudaGetErrorString musaGetErrorString
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#define cudaGetDevice musaGetDevice
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMemcpy musaMemcpy
#define cudaStreamCreate musaStreamCreate
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaEvent_t musaEvent_t
#define cudaEventCreate musaEventCreate
#define cudaEventDestroy musaEventDestroy
#define cudaEventElapsedTime musaEventElapsedTime
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize

using __nv_bfloat16 = __mt_bfloat16;
using __nv_bfloat162 = __mt_bfloat162;

__device__ __forceinline__ float2 musa_bfloat162_to_float2(
		__nv_bfloat162 value) {
	return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
}

#define __bfloat1622float2 musa_bfloat162_to_float2
