#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
// struct Header {
//     int64_t num_rows;
//     int64_t num_cols;
//     int32_t blocksize;
// };
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

template <typename T>
__device__ __forceinline__ T float_to_out(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_out<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ half float_to_out<half>(float v) {
    return __float2half(v);
}

template <typename T>
__device__ __forceinline__ uint32_t pack_pair_to_u32(float v1, float v2);

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<__nv_bfloat16>(float v1, float v2) {
    __nv_bfloat162 packed = __floats2bfloat162_rn(v1, v2);
    return *reinterpret_cast<uint32_t*>(&packed);
}

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<half>(float v1, float v2) {
    half2 packed = __floats2half2_rn(v1, v2);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// __global__ void nf4_decode_kernel_native(
//     const uint8_t* __restrict__ packed_weights,
//     const uint8_t* __restrict__ absmax_q,
//     const half* __restrict__ absmax2,
//     const half* __restrict__ code2,
//     const float offset, // 通常为 0
//     __nv_bfloat16* __restrict__ output,
//     int64_t num_elements,
//     int blocksize,
//     int group_size
// ) {
//     // 1. 全局一维线程索引
//     // 每个线程负责 1 个字节（即 2 个 4-bit 权重）
//     int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int64_t total_bytes = (num_elements + 1) / 2;
//     int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

//     for (int64_t byte_idx = tid; byte_idx < total_bytes; byte_idx += stride) {
//         // 2. 读取这 1 个字节，并解包成两个 4-bit 索引
//         uint8_t packed = packed_weights[byte_idx];
//         uint8_t idx1 = (packed >> 4) & 0x0F; // 高 4 位对应 output[byte_idx * 2]
//         uint8_t idx2 = packed & 0x0F;        // 低 4 位对应 output[byte_idx * 2 + 1]

//         // 3. 计算当前字节属于哪一个量化 Block 和 Group
//         // 每一个 Block 包含 blocksize 个权重，即 blocksize / 2 个字节
//         int bytes_per_block = blocksize / 2;
//         int block_id = static_cast<int>(byte_idx / bytes_per_block);
//         int group_id = block_id / group_size;

//         // 4. 暴力从全局内存 (Global Memory) 读取双重量化参数
//         float a2 = __half2float(absmax2[group_id]);
//         uint8_t qa = absmax_q[block_id];
//         float c2 = __half2float(code2[qa]);
//         float real_absmax = c2 * a2 + offset;

//         // 5. 结合 NF4 查表，计算真实的浮点权重
//         float w1_fp32 = NF4_LUT[idx1] * real_absmax;
//         float w2_fp32 = NF4_LUT[idx2] * real_absmax;

//         // 6. 分别写回 
//         int64_t out_idx = byte_idx * 2;
//         if (out_idx < num_elements) {
//             output[out_idx] = __float2bfloat16(w1_fp32);
//         }
//         if (out_idx + 1 < num_elements) {
//             output[out_idx + 1] = __float2bfloat16(w2_fp32);
//         }
//     }
// }

template <typename OutT>
__global__ void nf4_decode_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    const float offset,
    OutT* __restrict__ output,
    int64_t num_elements,
    int blocksize,
    int group_size
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_bytes = (num_elements + 1) / 2;
    int64_t full_pair_bytes = num_elements / 2;
    unsigned warp_mask = 0xffffffffu;
    int lane = threadIdx.x & 31;

    __shared__ float s_LUT[16];
    if (threadIdx.x < 16) {
        s_LUT[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);

    for (int64_t byte_idx = tid; byte_idx < full_pair_bytes; byte_idx += stride) {
        uint8_t packed = packed_weights[byte_idx];
        int block_id = static_cast<int>(byte_idx / (blocksize / 2));
        int group_id = block_id / group_size;
        float real_absmax = 0.0f;
        if (lane == 0) {
            uint8_t qa = absmax_q[block_id];
            real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[qa])) + offset;
        }
        real_absmax = __shfl_sync(warp_mask, real_absmax, 0);
        float v1 = s_LUT[packed >> 4] * real_absmax;
        float v2 = s_LUT[packed & 0x0F] * real_absmax;
        out_u32[byte_idx] = pack_pair_to_u32<OutT>(v1, v2);
    }

    if ((num_elements & 1) != 0) {
        int64_t tail_byte = total_bytes - 1;
        if (tid == 0) {
            uint8_t packed = packed_weights[tail_byte];
            int block_id = static_cast<int>(tail_byte / (blocksize / 2));
            int group_id = block_id / group_size;
            float real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]])) + offset;
            output[num_elements - 1] = float_to_out<OutT>(s_LUT[packed >> 4] * real_absmax);
        }
    }
}

int main(int argc, char** argv) {
//     1.输入解析，读取二进制文件
    std::string input_file = "03_nf4_dequant/ikko/data/weight_data.bin";
    std::string output_file = "03_nf4_dequant/ikko/data/output.bin";
    enum class OutputType { BF16, FP16 };
    OutputType output_type = OutputType::BF16;
    if (argc >= 2) {
        if (std::strcmp(argv[1], "bf16") == 0) {
            output_type = OutputType::BF16;
            output_file = "03_nf4_dequant/ikko/data/output_bf16.bin";
        } else if (std::strcmp(argv[1], "fp16") == 0) {
            output_type = OutputType::FP16;
            output_file = "03_nf4_dequant/ikko/data/output_fp16.bin";
        } else {
            std::cerr << "Usage: " << argv[0] << " [bf16|fp16] [output_file]" << std::endl;
            return 1;
        }
    }
    if (argc >= 3) {
        output_file = argv[2];
    }
    std::ifstream infile(input_file, std::ios::binary);
    if (!infile) {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            std::cerr << "CWD: " << cwd << std::endl;
        }
        std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
        std::string fallback_file = "03_nf4_dequant/ikko/data/weight_data.bin";
        infile.open(fallback_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open fallback input file: " << fallback_file << std::endl;
            return 1;
        }
        input_file = fallback_file;
    }

// 1. 读取 Header
    int64_t num_rows, num_cols;
    int32_t blocksize;
    infile.read(reinterpret_cast<char*>(&num_rows), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&num_cols), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&blocksize), sizeof(int32_t));
    std::streampos data_start = infile.tellg();
    infile.seekg(0, std::ios::end);
    int64_t file_size = static_cast<int64_t>(infile.tellg());
    infile.seekg(data_start, std::ios::beg);
// 2.内存规划
    // Header header{num_rows, num_cols, blocksize};
    int64_t num_elements = num_rows * num_cols;
    int64_t num_blocks = (num_elements + blocksize - 1) / blocksize;
    int64_t num_groups = 0;

// 3.数据加载，分配显存
    size_t size_packed = num_elements >> 1; // 需要 num_elements / 2 个字节，一个 byte 存 2 个权重
    size_t size_packed_padded = (size_packed + 15) & ~static_cast<size_t>(15);
    size_t size_absmax_q = num_blocks * sizeof(uint8_t);
    size_t size_absmax2 = 0; // float16
    size_t size_code2 = 256 * sizeof(half); // float16
    float offset; // float32

    std::vector<uint8_t> h_packed(size_packed_padded, 0);
    std::vector<uint8_t> h_absmax_q(num_blocks);
    // 推断 num_groups（兼容不同 bitsandbytes 版本的分组大小）
    int64_t header_size = static_cast<int64_t>(sizeof(int64_t) * 2 + sizeof(int32_t));
    int64_t remaining = file_size - header_size -
                        static_cast<int64_t>(size_packed + size_absmax_q + size_code2 + sizeof(float));
    if (remaining > 0 && (remaining % sizeof(half) == 0)) {
        num_groups = remaining / static_cast<int64_t>(sizeof(half));
        size_absmax2 = static_cast<size_t>(num_groups) * sizeof(half);
    } else {
        num_groups = (num_blocks + 255) / 256;
        size_absmax2 = static_cast<size_t>(num_groups) * sizeof(half);
    }

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
    __nv_bfloat16 *d_output_bf16 = nullptr;
    half *d_output_fp16 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed, size_packed_padded));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, size_absmax_q));
    CHECK_CUDA(cudaMalloc(&d_absmax2, size_absmax2));
    CHECK_CUDA(cudaMalloc(&d_code2, size_code2));
    if (output_type == OutputType::BF16) {
        CHECK_CUDA(cudaMalloc(&d_output_bf16, num_elements * sizeof(__nv_bfloat16)));
    } else {
        CHECK_CUDA(cudaMalloc(&d_output_fp16, num_elements * sizeof(half)));
    }
    CHECK_CUDA(cudaMemcpy(d_packed, h_packed.data(), size_packed_padded, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), size_absmax_q, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, h_absmax2.data(), size_absmax2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2, h_code2.data(), size_code2, cudaMemcpyHostToDevice));
    


// 4. 启动 CUDA Kernel
    dim3 blockDim(256);
    int64_t total_bytes = (num_elements + 1) / 2;
    int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int max_active_blocks = 0;
    if (output_type == OutputType::BF16) {
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            nf4_decode_kernel<__nv_bfloat16>,
            blockDim.x,
            0));
    } else {
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            nf4_decode_kernel<half>,
            blockDim.x,
            0));
    }
    int grid_x = sm_count * max_active_blocks;
    int64_t max_grid = (total_bytes + blockDim.x - 1) / blockDim.x;
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
    if (output_type == OutputType::BF16) {
        nf4_decode_kernel<__nv_bfloat16><<<gridDim, blockDim>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output_bf16, num_elements, blocksize, group_size
        );
    } else {
        nf4_decode_kernel<half><<<gridDim, blockDim>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output_fp16, num_elements, blocksize, group_size
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (output_type == OutputType::BF16) {
            nf4_decode_kernel<__nv_bfloat16><<<gridDim, blockDim>>>(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output_bf16, num_elements, blocksize, group_size
            );
        } else {
            nf4_decode_kernel<half><<<gridDim, blockDim>>>(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output_fp16, num_elements, blocksize, group_size
            );
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
// 5.记录性能，写入数据
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iters;

// 6. D2H 拷贝结果
    std::vector<__nv_bfloat16> h_output_bf16;
    std::vector<half> h_output_fp16;
    if (output_type == OutputType::BF16) {
        h_output_bf16.resize(num_elements);
        CHECK_CUDA(cudaMemcpy(h_output_bf16.data(), d_output_bf16, num_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    } else {
        h_output_fp16.resize(num_elements);
        CHECK_CUDA(cudaMemcpy(h_output_fp16.data(), d_output_fp16, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    }

// 7. 计算并打印性能
    double total_io_bytes = static_cast<double>(size_packed + size_absmax_q + size_absmax2 + size_code2) +
                            static_cast<double>(num_elements * 2);
    double bandwidth = total_io_bytes / (milliseconds / 1000.0) / 1e9;
    constexpr double bnb_ref_ms = 1.243360;
    constexpr double bnb_ref_bw = 543.14;
    double speedup_vs_bnb = bnb_ref_ms / static_cast<double>(milliseconds);
    double bw_ratio_vs_bnb = bandwidth / bnb_ref_bw;
    std::cout << "Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth (approx): " << bandwidth << " GB/s" << std::endl;
    std::cout << "Speedup vs bitsandbytes: " << speedup_vs_bnb << "x"
              << " (ref " << bnb_ref_ms << " ms)" << std::endl;
    std::cout << "Bandwidth ratio vs bitsandbytes: " << bw_ratio_vs_bnb << "x"
              << " (ref " << bnb_ref_bw << " GB/s)" << std::endl;
    std::cout << "Output dtype: " << (output_type == OutputType::BF16 ? "bf16" : "fp16") << std::endl;

// 8. 写入输出文件
    std::ofstream outfile(output_file, std::ios::binary);
    if (output_type == OutputType::BF16) {
        outfile.write(reinterpret_cast<char*>(h_output_bf16.data()), num_elements * sizeof(__nv_bfloat16));
    } else {
        outfile.write(reinterpret_cast<char*>(h_output_fp16.data()), num_elements * sizeof(half));
    }
    outfile.close();
    std::cout << "Output written to " << output_file << std::endl;

    // 清理
    cudaFree(d_packed);
    cudaFree(d_absmax_q);
    cudaFree(d_absmax2);
    cudaFree(d_code2);
    if (d_output_bf16) cudaFree(d_output_bf16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}