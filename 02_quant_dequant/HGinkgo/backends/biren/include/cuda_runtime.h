#pragma once

#include <supa.h>

using cudaError_t = suError_t;
using cudaEvent_t = suEvent_t;
using cudaDeviceProp = suDeviceProp;
using cudaMemcpyKind = suMemcpyKind;

constexpr cudaError_t cudaSuccess = suSuccess;
constexpr cudaError_t cudaErrorInsufficientDriver = suErrorInsufficientDriver;
constexpr cudaError_t cudaErrorNoDevice = suErrorNoDevice;
constexpr cudaMemcpyKind cudaMemcpyHostToHost = suMemcpyHostToHost;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = suMemcpyHostToDevice;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = suMemcpyDeviceToHost;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = suMemcpyDeviceToDevice;
constexpr cudaMemcpyKind cudaMemcpyDefault = suMemcpyDefault;

#define cudaDeviceSynchronize suDeviceSynchronize
#define cudaEventCreate suEventCreate
#define cudaEventDestroy suEventDestroy
#define cudaEventElapsedTime suEventElapsedTime
#define cudaEventRecord suEventRecord
#define cudaEventSynchronize suEventSynchronize
#define cudaFree suFree
#define cudaGetDeviceCount suGetDeviceCount
#define cudaGetDeviceProperties suGetDeviceProperties
#define cudaGetErrorString suGetErrorString
#define cudaGetLastError suGetLastError
#define cudaMalloc suMallocDevice
#define cudaMemcpy suMemcpy
#define cudaMemset suMemset
#define cudaSetDevice suSetDevice
