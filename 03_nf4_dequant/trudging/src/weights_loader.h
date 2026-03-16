#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#if defined(PLATFORM_METAX)
    #include <mcr/mc_runtime.h>
    
    #define CUDA_MALLOC_HOST mcMallocHost
    #define CUDA_FREE_HOST mcFreeHost
    #define CUDA_SUCCESS mcSuccess
    #define CUDA_GET_ERROR_STRING mcGetErrorString
    #define CUDA_ERROR_T mcError_t
#elif defined(PLATFORM_MOORE)
    #include <musa_runtime.h>
    #define CUDA_MALLOC_HOST musaMallocHost
    #define CUDA_FREE_HOST musaFreeHost
    #define CUDA_SUCCESS musaSuccess
    #define CUDA_GET_ERROR_STRING musaGetErrorString
    #define CUDA_ERROR_T musaError_t
#else
    #include <cuda_runtime.h>
    #define CUDA_MALLOC_HOST cudaMallocHost
    #define CUDA_FREE_HOST cudaFreeHost
    #define CUDA_SUCCESS cudaSuccess
    #define CUDA_GET_ERROR_STRING cudaGetErrorString
    #define CUDA_ERROR_T cudaError_t
#endif

// Custom deleter
struct CudaHostDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            CUDA_FREE_HOST(ptr);
        }
    }
};

// 别名定义，方便使用
template <typename T>
using start_pinned_ptr = std::unique_ptr<T[], CudaHostDeleter>;

// 辅助函数：分配 pinned memory
template <typename T>
start_pinned_ptr<T> allocate_pinned(size_t count) {
    void* ptr = nullptr;
    CUDA_ERROR_T err = CUDA_MALLOC_HOST(&ptr, count * sizeof(T));
    if (err != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("CUDA_MALLOC_HOST failed: ") + CUDA_GET_ERROR_STRING(err));
    }
    return start_pinned_ptr<T>(static_cast<T*>(ptr));
}

struct QuantizedWeights {
    int64_t num_rows;
    int64_t num_cols;
    int32_t block_size;

    size_t num_blocks;
    size_t num_groups;
    size_t packed_size;

    // 使用智能指针管理的 Pinned Memory 数组
    start_pinned_ptr<uint8_t> packed_weights;
    start_pinned_ptr<uint8_t> absmax_q;
    start_pinned_ptr<uint16_t> absmax2;
    start_pinned_ptr<uint16_t> code2;
    
    float offset; // 单个 float 值
};

inline QuantizedWeights load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    QuantizedWeights w;

    // 1. 读取头部
    if (!file.read(reinterpret_cast<char*>(&w.num_rows), sizeof(w.num_rows))) throw std::runtime_error("Failed to read num_rows");
    if (!file.read(reinterpret_cast<char*>(&w.num_cols), sizeof(w.num_cols))) throw std::runtime_error("Failed to read num_cols");
    if (!file.read(reinterpret_cast<char*>(&w.block_size), sizeof(w.block_size))) throw std::runtime_error("Failed to read block_size");

    // 2. 计算各部分大小
    // 注意：这里假设 num_rows * num_cols 是偶数，或者按照 (N*M)/2 向下取整。
    // 如果是 4-bit 量化，通常你需要确保总元素个数是偶数，或者处理尾部 padding。
    w.packed_size = (w.num_rows * w.num_cols) / 2;
    
    // num_blocks = ceil(num_rows * num_cols / blocksize)
    w.num_blocks = (w.num_rows * w.num_cols + w.block_size - 1) / w.block_size;
    
    // num_groups = ceil(num_blocks / 256)
    // 根据您的要求：block_size_2 为固定 256
    // 注：原问题中提到 "absmax2: ... 长度为 num_groups (假设固定为 256)" 
    // 但后续追问指出应为计算值。此处按追问逻辑计算 num_groups。
    // 如果 "假设固定为 256" 指的是 group_size，则如下计算：
    size_t group_size = 256;
    w.num_groups = (w.num_blocks + group_size - 1) / group_size;

    // 3. 分配 Pinned Memory
    try {
        w.packed_weights = allocate_pinned<uint8_t>(w.packed_size);
        w.absmax_q = allocate_pinned<uint8_t>(w.num_blocks);
        w.absmax2 = allocate_pinned<uint16_t>(w.num_groups);
        w.code2 = allocate_pinned<uint16_t>(256); // 固定 256 元素
    } catch (const std::exception& e) {
        file.close();
        throw;
    }

    // 4. 读取数据数组
    auto read_array = [&](char* dst, size_t size, const char* name) {
        file.read(dst, size);
        if (file.gcount() != static_cast<std::streamsize>(size)) {
            throw std::runtime_error(std::string("Failed to read ") + name + ". Expected " + std::to_string(size) + " bytes, got " + std::to_string(file.gcount()));
        }
    };

    read_array(reinterpret_cast<char*>(w.packed_weights.get()), w.packed_size * sizeof(uint8_t), "packed_weights");
    read_array(reinterpret_cast<char*>(w.absmax_q.get()), w.num_blocks * sizeof(uint8_t), "absmax_q");
    read_array(reinterpret_cast<char*>(w.absmax2.get()), w.num_groups * sizeof(uint16_t), "absmax2");
    read_array(reinterpret_cast<char*>(w.code2.get()), 256 * sizeof(uint16_t), "code2");

    // 5. 读取 offset
    if (!file.read(reinterpret_cast<char*>(&w.offset), sizeof(w.offset))) {
        throw std::runtime_error("Failed to read offset");
    }

    // 6. 检查是否还有剩余数据（可选，视文件格式严格程度而定）
    if (file.peek() != EOF) {
        std::cerr << "Warning: Extra data found at the end of the file " << filename << std::endl;
    }

    file.close();
    return w;
}
