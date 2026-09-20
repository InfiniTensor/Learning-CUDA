#pragma once

// MXMACA 兼容层：把当前工程使用的 CUDA runtime 名称映射到 MXMACA
// runtime。该文件只应在 MXMACA 构建中通过 metax_include 路径启用，
// 不改变 NVIDIA 构建和文件格式契约。
#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_fp16.hpp>
#include <common/maca_bfloat16.h>
#include <common/maca_bfloat16.hpp>

using __nv_bfloat16 = __maca_bfloat16;
using __nv_bfloat162 = __maca_bfloat162;

#define cudaError_t mcError_t
#define cudaSuccess mcSuccess
#define cudaErrorNoDevice mcErrorNoDevice
#define cudaGetErrorString mcGetErrorString
#define cudaGetLastError mcGetLastError
#define cudaDeviceSynchronize mcDeviceSynchronize
#define cudaGetDeviceCount mcGetDeviceCount
#define cudaGetDevice mcGetDevice
#define cudaGetDeviceProperties mcGetDeviceProperties
#define cudaRuntimeGetVersion mcRuntimeGetVersion
#define cudaDriverGetVersion mcDriverGetVersion

#define cudaDeviceProp mcDeviceProp_t
#define cudaEvent_t mcEvent_t
#define cudaEventCreate mcEventCreate
#define cudaEventDestroy mcEventDestroy
#define cudaEventElapsedTime mcEventElapsedTime
#define cudaEventRecord mcEventRecord
#define cudaEventSynchronize mcEventSynchronize

#define cudaMalloc mcMalloc
#define cudaFree mcFree
#define cudaMemcpy mcMemcpy
#define cudaMemcpyKind mcMemcpyKind
#define cudaMemcpyDeviceToHost mcMemcpyDeviceToHost
#define cudaMemcpyHostToDevice mcMemcpyHostToDevice
