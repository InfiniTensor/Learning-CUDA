#pragma once

// 仅在 MUSA 构建中加入此目录：公共 runtime 使用原接口名，实际链接 musart。
#include <musa_runtime.h>
#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMemcpy musaMemcpy
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaEvent_t musaEvent_t
#define cudaEventCreate musaEventCreate
#define cudaEventDestroy musaEventDestroy
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEventElapsedTime musaEventElapsedTime
