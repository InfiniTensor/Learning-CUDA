#pragma once

#include "low_precision/ascend_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

using cudaError_t = aclError;
using cudaEvent_t = aclrtEvent;
using cudaMemcpyKind = aclrtMemcpyKind;

struct cudaDeviceProp {
    char name[256] = "Ascend";
};

constexpr cudaError_t cudaSuccess = ACL_SUCCESS;
constexpr cudaError_t cudaErrorInsufficientDriver = static_cast<cudaError_t>(-1);
constexpr cudaError_t cudaErrorNoDevice = static_cast<cudaError_t>(-2);
constexpr cudaMemcpyKind cudaMemcpyHostToHost = ACL_MEMCPY_HOST_TO_HOST;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;
constexpr cudaMemcpyKind cudaMemcpyDefault = ACL_MEMCPY_DEFAULT;

inline cudaError_t cudaGetDeviceCount(int* count) {
    const cudaError_t status = low_precision::ascend::initialize();
    if (status != cudaSuccess) {
        return status;
    }
    std::uint32_t device_count = 0;
    const cudaError_t result = aclrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return result;
}

inline cudaError_t cudaSetDevice(int device) {
    return device == 0 ? low_precision::ascend::initialize()
                       : static_cast<cudaError_t>(ACL_ERROR_RT_PARAM_INVALID);
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
    if (device != 0) {
        return static_cast<cudaError_t>(ACL_ERROR_RT_PARAM_INVALID);
    }
    std::strncpy(properties->name, "Ascend", sizeof(properties->name));
    return low_precision::ascend::initialize();
}

inline const char* cudaGetErrorString(cudaError_t status) {
    return low_precision::ascend::error_string(status);
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** pointer, std::size_t size) {
    const cudaError_t status = low_precision::ascend::initialize();
    return status == cudaSuccess ? aclrtMalloc(pointer, size, ACL_MEM_MALLOC_HUGE_FIRST) : status;
}

template <typename T>
inline cudaError_t cudaMalloc(T** pointer, std::size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(pointer), size);
}

inline cudaError_t cudaFree(void* pointer) {
    return pointer == nullptr ? cudaSuccess : aclrtFree(pointer);
}

inline cudaError_t cudaMemcpy(void* destination,
                              const void* source,
                              std::size_t size,
                              cudaMemcpyKind kind) {
    const cudaError_t status = low_precision::ascend::initialize();
    if (status != cudaSuccess) {
        return status;
    }
    if (kind == cudaMemcpyHostToHost) {
        std::memcpy(destination, source, size);
        return cudaSuccess;
    }
    return aclrtMemcpy(destination, size, source, size, kind);
}

inline cudaError_t cudaMemset(void* pointer, int value, std::size_t size) {
    const cudaError_t status = low_precision::ascend::initialize();
    return status == cudaSuccess ? aclrtMemset(pointer, size, value, size) : status;
}

inline cudaError_t cudaDeviceSynchronize() {
    const cudaError_t status = low_precision::ascend::initialize();
    return status == cudaSuccess ? aclrtSynchronizeStream(low_precision::ascend::stream()) : status;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    const cudaError_t status = low_precision::ascend::initialize();
    return status == cudaSuccess ? aclrtCreateEvent(event) : status;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return aclrtDestroyEvent(event);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event) {
    const cudaError_t status = low_precision::ascend::initialize();
    return status == cudaSuccess ? aclrtRecordEvent(event, low_precision::ascend::stream()) : status;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return aclrtSynchronizeEvent(event);
}

inline cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t start, cudaEvent_t stop) {
    return aclrtEventElapsedTime(milliseconds, start, stop);
}
