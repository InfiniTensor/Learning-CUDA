#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
struct Header {
    int64_t num_rows;
    int64_t num_cols;
    int32_t blocksize;
};
__constant__ float NF4_LUT[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};
void checkCuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << std::endl;
        exit(99);
    }
}
#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

__global__ void nf4_decode_kernel_native(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    const float offset, // 通常为 0
    __nv_bfloat16* __restrict__ output,
    int64_t num_elements,
    int blocksize
) {
    // 1. 全局一维线程索引
    // 每个线程负责 1 个字节（即 2 个 4-bit 权重）
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = num_elements / 2;

    // 边界检查：多余的线程直接退出
    if (tid >= total_bytes) return;

    // 2. 读取这 1 个字节，并解包成两个 4-bit 索引
    uint8_t packed = packed_weights[tid];
    uint8_t idx1 = packed & 0x0F;
    uint8_t idx2 = (packed >> 4) & 0x0F;

    // 3. 计算当前字节属于哪一个量化 Block 和 Group
    // 每一个 Block 包含 blocksize 个权重，即 blocksize / 2 个字节
    int bytes_per_block = blocksize / 2; 
    int block_id = tid / bytes_per_block; // 当前 byte 属于第几个 64-weight block
    int group_id = block_id / 256; // bitsandbytes 默认每组 256 个 block，256 个 block 共享一个 absmax2

    // 4. 从全局内存 (Global Memory) 读取双重量化参数
    float a2 = __half2float(absmax2[group_id]);   // 读取二级缩放
    uint8_t qa = absmax_q[block_id];              // 读取一级缩放索引
    float c2 = __half2float(code2[qa]);           // 查码表解码一级缩放
    float real_absmax = c2 * a2;                  // 计算最终缩放因子

    // 5. 结合 NF4 查表，计算真实的浮点权重
    float w1_fp32 = NF4_LUT[idx1] * real_absmax + offset;
    float w2_fp32 = NF4_LUT[idx2] * real_absmax + offset;

    // 6. 最朴素的分别写回 
    output[tid * 2]     = __float2bfloat16(w1_fp32);
    output[tid * 2 + 1] = __float2bfloat16(w2_fp32);
}

__global__ void nf4_decode_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    const float offset, // 通常为 0
    __nv_bfloat16* __restrict__ output,
    int64_t num_elements,
    int blocksize
) {

}



int main(int argc, char** argv) {
//     1.输入解析，读取二进制文件
    std::string input_file = "test_data/weight_data.bin";
    std::string output_file = "test_data/output.bin";
    std::ifstream infile(input_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open input file." << std::endl;
        return 1;
    }

// 1. 读取 Header
    int64_t num_rows, num_cols;
    int32_t blocksize;
    infile.read(reinterpret_cast<char*>(&num_rows), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&num_cols), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&blocksize), sizeof(int32_t));
// 2.内存规划
    Header header{num_rows, num_cols, blocksize};
    int64_t num_elements = num_rows * num_cols;
    int64_t num_blocks = (num_elements + blocksize - 1) / blocksize;
    int64_t num_groups = (num_blocks + 255) / 256; // 每个 block 256 个线程

// 3.数据加载，分配显存
    size_t size_packed = num_elements >>1; //需要 num_elements / 2 个字节，一个 byte 存 2 个权重
    size_t size_absmax_q = num_blocks * sizeof(uint8_t);
    size_t size_absmax2 = num_groups * sizeof(half); // float16
    size_t size_code2 = 256 * sizeof(half); // float16
    float offset; // float32

    std::vector<uint8_t> h_packed(size_packed);
    std::vector<uint8_t> h_absmax_q(num_blocks);
    std::vector<half> h_absmax2(num_groups);
    std::vector<half> h_code2(256);

    infile.read(reinterpret_cast<char*>(h_packed.data()), size_packed);
    infile.read(reinterpret_cast<char*>(h_absmax_q.data()), size_absmax_q);
    infile.read(reinterpret_cast<char*>(h_absmax2.data()), size_absmax2);
    infile.read(reinterpret_cast<char*>(h_code2.data()), size_code2);
    infile.read(reinterpret_cast<char*>(&offset), sizeof(float)); 

    infile.close();
    // 分配 device   内存
    uint8_t* d_packed = nullptr;
    uint8_t* d_absmax_q = nullptr;
    half* d_absmax2 = nullptr;
    half* d_code2 = nullptr;
    __nv_bfloat16 *d_output;

    CHECK_CUDA(cudaMalloc(&d_packed, size_packed));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, size_absmax_q));
    CHECK_CUDA(cudaMalloc(&d_absmax2, size_absmax2));
    CHECK_CUDA(cudaMalloc(&d_code2, size_code2));
    CHECK_CUDA(cudaMalloc(&d_output, num_elements * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_packed, h_packed.data(), size_packed, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), size_absmax_q, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, h_absmax2.data(), size_absmax2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2, h_code2.data(), size_code2, cudaMemcpyHostToDevice));
    


// 4. 启动 CUDA Kernel
    dim3 blockDim(256);
    dim3 blockDim(256);
    int64_t total_bytes = (num_elements + 1) / 2;
    int64_t total_words = (total_bytes + 15) / 16;
    int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int max_active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        nf4_decode_kernel,
        blockDim.x,
        0));
    int grid_x = sm_count * max_active_blocks;
    int64_t max_grid = (total_words + blockDim.x - 1) / blockDim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 gridDim(grid_x);
    std::cout << "SM count: " << sm_count
              << ", max active blocks/SM: " << max_active_blocks
              << ", grid_x: " << grid_x << std::endl;
    int group_size = static_cast<int>((num_blocks + num_groups - 1) / num_groups);
    // kernel 函数需要完成 NF4 解码的核心计算逻辑
    // 计时事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    nf4_decode_kernel<<<gridDim, blockDim>>>(
        d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output, num_elements, blocksize
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        nf4_decode_kernel<<<gridDim, blockDim>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output, num_elements, blocksize
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
// 5.记录性能，写入数据
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iters;

// 6. D2H 拷贝结果
    std::vector<__nv_bfloat16> h_output(num_elements);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

// 7. 计算并打印性能
    double total_bytes = static_cast<double>(size_packed + size_absmax_q + size_absmax2 + size_code2) +
                         static_cast<double>(num_elements * 2);
    double bandwidth = total_bytes / (milliseconds / 1000.0) / 1e9;
    std::cout << "Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth (approx): " << bandwidth << " GB/s" << std::endl;

// 8. 写入输出文件
    std::ofstream outfile(output_file, std::ios::binary);
    outfile.write(reinterpret_cast<char*>(h_output.data()), num_elements * sizeof(__nv_bfloat16));
    outfile.close();
    std::cout << "Output written to " << output_file << std::endl;

    // 清理
    cudaFree(d_packed);
    cudaFree(d_absmax_q);
    cudaFree(d_absmax2);
    cudaFree(d_code2);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}