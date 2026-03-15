#include <musa_runtime.h>
#include <musa_fp16.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <unistd.h>

__constant__ float NF4_LUT[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

void checkMusa(musaError_t result, const char* func, const char* file, int line) {
    if (result != musaSuccess) {
        std::cerr << "MUSA error at " << file << ":" << line << " code=" << result << " \"" << func << "\"\n";
        std::cerr << "Error string: " << musaGetErrorString(result) << std::endl;
        std::exit(99);
    }
}

#define CHECK_MUSA(val) checkMusa((val), #val, __FILE__, __LINE__)

__device__ __forceinline__ uint32_t pack_pair_to_u32(float v1, float v2) {
    half2 packed = __floats2half2_rn(v1, v2);
    return *reinterpret_cast<uint32_t*>(&packed);
}

__global__ void nf4_decode_kernel_fp16(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    float offset,
    half* __restrict__ output,
    int64_t num_elements,
    int blocksize,
    int group_size) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t total_bytes = (num_elements + 1) / 2;
    int64_t full_pair_bytes = num_elements / 2;
    int bytes_per_block = blocksize / 2;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();

    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);
    for (int64_t byte_idx = tid; byte_idx < full_pair_bytes; byte_idx += stride) {
        uint8_t packed = packed_weights[byte_idx];
        int block_id = static_cast<int>(byte_idx / bytes_per_block);
        int group_id = block_id / group_size;

        float real_absmax = __half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]]) + offset;
        float v1 = s_lut[packed >> 4] * real_absmax;
        float v2 = s_lut[packed & 0x0F] * real_absmax;
        out_u32[byte_idx] = pack_pair_to_u32(v1, v2);
    }

    if ((num_elements & 1) != 0 && tid == 0) {
        int64_t tail_byte = total_bytes - 1;
        uint8_t packed = packed_weights[tail_byte];
        int block_id = static_cast<int>(tail_byte / bytes_per_block);
        int group_id = block_id / group_size;
        float real_absmax = __half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]]) + offset;
        output[num_elements - 1] = __float2half(s_lut[packed >> 4] * real_absmax);
    }
}

int main(int argc, char** argv) {
    std::string input_file = "03_nf4_dequant/ikko/data/weight_data.bin";
    std::string output_file = "03_nf4_dequant/ikko/output_fp16_mu.bin";

    if (argc >= 2) {
        if (std::strcmp(argv[1], "fp16") != 0) {
            std::cerr << "Warning: mainla.mu currently supports fp16 only, got '" << argv[1]
                      << "'. Continue with fp16 output." << std::endl;
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

    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int32_t blocksize = 0;
    infile.read(reinterpret_cast<char*>(&num_rows), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&num_cols), sizeof(int64_t));
    infile.read(reinterpret_cast<char*>(&blocksize), sizeof(int32_t));

    std::streampos data_start = infile.tellg();
    infile.seekg(0, std::ios::end);
    int64_t file_size = static_cast<int64_t>(infile.tellg());
    infile.seekg(data_start, std::ios::beg);

    int64_t num_elements = num_rows * num_cols;
    int64_t num_blocks = (num_elements + blocksize - 1) / blocksize;

    size_t size_packed = static_cast<size_t>((num_elements + 1) / 2);
    size_t size_packed_padded = (size_packed + 15) & ~static_cast<size_t>(15);
    size_t size_absmax_q = static_cast<size_t>(num_blocks) * sizeof(uint8_t);
    size_t size_code2 = 256 * sizeof(half);
    size_t size_absmax2 = 0;
    int64_t num_groups = 0;

    int64_t header_size = static_cast<int64_t>(sizeof(int64_t) * 2 + sizeof(int32_t));
    int64_t remaining = file_size - header_size -
                        static_cast<int64_t>(size_packed + size_absmax_q + size_code2 + sizeof(float));
    if (remaining > 0 && (remaining % static_cast<int64_t>(sizeof(half)) == 0)) {
        num_groups = remaining / static_cast<int64_t>(sizeof(half));
    } else {
        num_groups = (num_blocks + 255) / 256;
    }
    size_absmax2 = static_cast<size_t>(num_groups) * sizeof(half);

    std::vector<uint8_t> h_packed(size_packed_padded, 0);
    std::vector<uint8_t> h_absmax_q(static_cast<size_t>(num_blocks));
    std::vector<half> h_absmax2(static_cast<size_t>(num_groups));
    std::vector<half> h_code2(256);
    float offset = 0.0f;

    infile.read(reinterpret_cast<char*>(h_packed.data()), size_packed);
    infile.read(reinterpret_cast<char*>(h_absmax_q.data()), size_absmax_q);
    infile.read(reinterpret_cast<char*>(h_absmax2.data()), size_absmax2);
    infile.read(reinterpret_cast<char*>(h_code2.data()), size_code2);
    infile.read(reinterpret_cast<char*>(&offset), sizeof(float));
    infile.close();

    uint8_t* d_packed = nullptr;
    uint8_t* d_absmax_q = nullptr;
    half* d_absmax2 = nullptr;
    half* d_code2 = nullptr;
    half* d_output = nullptr;

    CHECK_MUSA(musaMalloc(&d_packed, size_packed_padded));
    CHECK_MUSA(musaMalloc(&d_absmax_q, size_absmax_q));
    CHECK_MUSA(musaMalloc(&d_absmax2, size_absmax2));
    CHECK_MUSA(musaMalloc(&d_code2, size_code2));
    CHECK_MUSA(musaMalloc(&d_output, static_cast<size_t>(num_elements) * sizeof(half)));

    CHECK_MUSA(musaMemcpy(d_packed, h_packed.data(), size_packed_padded, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_absmax_q, h_absmax_q.data(), size_absmax_q, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_absmax2, h_absmax2.data(), size_absmax2, musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_code2, h_code2.data(), size_code2, musaMemcpyHostToDevice));

    dim3 block_dim(256);
    int64_t total_bytes = (num_elements + 1) / 2;
    int grid_x = 4096;
    int64_t max_grid = (total_bytes + block_dim.x - 1) / block_dim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 grid_dim(grid_x);

    int group_size = static_cast<int>((num_blocks + num_groups - 1) / num_groups);

    nf4_decode_kernel_fp16<<<grid_dim, block_dim>>>(
        d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output, num_elements, blocksize, group_size);
    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());

    const int iters = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        nf4_decode_kernel_fp16<<<grid_dim, block_dim>>>(
            d_packed, d_absmax_q, d_absmax2, d_code2, offset, d_output, num_elements, blocksize, group_size);
    }
    CHECK_MUSA(musaGetLastError());
    CHECK_MUSA(musaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);

    std::vector<half> h_output(static_cast<size_t>(num_elements));
    CHECK_MUSA(musaMemcpy(h_output.data(), d_output, static_cast<size_t>(num_elements) * sizeof(half), musaMemcpyDeviceToHost));

    double total_io_bytes = static_cast<double>(size_packed + size_absmax_q + size_absmax2 + size_code2) +
                            static_cast<double>(num_elements * sizeof(half));
    double bandwidth = total_io_bytes / (milliseconds / 1000.0) / 1e9;

    std::cout << "grid_x: " << grid_x << std::endl;
    std::cout << "Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth (approx): " << bandwidth << " GB/s" << std::endl;
    std::cout << "Output dtype: fp16" << std::endl;

    std::ofstream outfile(output_file, std::ios::binary);
    outfile.write(reinterpret_cast<char*>(h_output.data()), static_cast<std::streamsize>(num_elements * sizeof(half)));
    outfile.close();
    std::cout << "Output written to " << output_file << std::endl;

    CHECK_MUSA(musaFree(d_packed));
    CHECK_MUSA(musaFree(d_absmax_q));
    CHECK_MUSA(musaFree(d_absmax2));
    CHECK_MUSA(musaFree(d_code2));
    CHECK_MUSA(musaFree(d_output));

    return 0;
}
