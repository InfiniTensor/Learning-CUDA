// NF4 反量化 CUDA 程序
// 用法: ./nf4_dequant <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>

#include "nf4_dequant_kernel.cuh"

// CUDA 错误检查
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// 二进制权重文件布局: header (rows, cols, blocksize) + packed_weights + absmax_q + absmax2 + code2 + offset
struct NF4Data {
    int64_t num_rows;
    int64_t num_cols;
    int32_t blocksize;

    std::vector<uint8_t> packed_weights;
    std::vector<uint8_t> absmax_q;
    std::vector<uint16_t> absmax2;    // fp16 raw bits
    std::vector<uint16_t> code2;      // fp16[256] raw bits
    float offset;

    int64_t n_elements;
    int32_t num_blocks;
    int32_t num_groups;
    int32_t s2_blocksize;
};

bool read_nf4_data(const char* filepath, NF4Data& data) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open file: %s\n", filepath);
        return false;
    }

    // Header
    fread(&data.num_rows, sizeof(int64_t), 1, f);
    fread(&data.num_cols, sizeof(int64_t), 1, f);
    fread(&data.blocksize, sizeof(int32_t), 1, f);

    data.n_elements = data.num_rows * data.num_cols;
    data.num_blocks = (int32_t)((data.n_elements + data.blocksize - 1) / data.blocksize);

    int64_t packed_size = data.n_elements / 2;
    data.packed_weights.resize(packed_size);
    fread(data.packed_weights.data(), 1, packed_size, f);

    data.absmax_q.resize(data.num_blocks);
    fread(data.absmax_q.data(), 1, data.num_blocks, f);

    // 从剩余字节反推 num_groups（文件中未显式存储）
    long current_pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, current_pos, SEEK_SET);

    long remaining = file_size - current_pos;
    long fixed_tail = 256 * 2 + 4;              // code2 (512B) + offset (4B)
    long absmax2_bytes = remaining - fixed_tail;
    data.num_groups = (int32_t)(absmax2_bytes / 2);
    data.s2_blocksize = (data.num_blocks + data.num_groups - 1) / data.num_groups;

    data.absmax2.resize(data.num_groups);
    fread(data.absmax2.data(), 2, data.num_groups, f);

    data.code2.resize(256);
    fread(data.code2.data(), 2, 256, f);

    fread(&data.offset, sizeof(float), 1, f);

    fclose(f);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]\n", argv[0]);
        return 1;
    }

    const char* weight_file = argv[1];
    const char* output_file = argv[2];
    std::string compute_type = (argc > 3) ? argv[3] : "bf16";
    int warmup = (argc > 4) ? atoi(argv[4]) : 10;
    int repeats = (argc > 5) ? atoi(argv[5]) : 100;

    bool use_bf16 = (compute_type == "bf16");

    // 读取数据
    printf("[INFO] 读取权重文件: %s\n", weight_file);
    NF4Data data;
    if (!read_nf4_data(weight_file, data)) return 1;

    printf("  num_rows     = %ld\n", (long)data.num_rows);
    printf("  num_cols     = %ld\n", (long)data.num_cols);
    printf("  blocksize    = %d\n", data.blocksize);
    printf("  n_elements   = %ld\n", (long)data.n_elements);
    printf("  num_blocks   = %d\n", data.num_blocks);
    printf("  num_groups   = %d\n", data.num_groups);
    printf("  s2_blocksize = %d\n", data.s2_blocksize);
    printf("  offset       = %f\n", data.offset);
    printf("  compute_type = %s\n", compute_type.c_str());

    // 分配 GPU 内存
    uint8_t* d_packed_weights;
    uint8_t* d_absmax_q;
    half* d_absmax2;
    half* d_code2;
    void* d_output;

    int64_t packed_size = data.n_elements / 2;
    int64_t output_bytes = data.n_elements * 2;  // bf16/fp16 = 2 bytes each

    CUDA_CHECK(cudaMalloc(&d_packed_weights, packed_size));
    CUDA_CHECK(cudaMalloc(&d_absmax_q, data.num_blocks));
    CUDA_CHECK(cudaMalloc(&d_absmax2, data.num_groups * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_code2, 256 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // H2D 传输
    CUDA_CHECK(cudaMemcpy(d_packed_weights, data.packed_weights.data(),
                          packed_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax_q, data.absmax_q.data(),
                          data.num_blocks, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax2, data.absmax2.data(),
                          data.num_groups * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code2, data.code2.data(),
                          256 * sizeof(half), cudaMemcpyHostToDevice));

    // Kernel launch 配置
    int n_packed = (int)((data.n_elements + 1) / 2);
    int n_packed_vec = (n_packed + 3) / 4;  // 每线程 4 字节
    int threads_per_block = 256;
    int num_blocks_kernel = (n_packed_vec + threads_per_block - 1) / threads_per_block;

    // 预计算 log2 用于位移优化
    int log2_bs = log2_pow2(data.blocksize);
    int log2_s2 = log2_pow2(data.s2_blocksize);

    printf("\n[INFO] Kernel 配置:\n");
    printf("  n_packed          = %d\n", n_packed);
    printf("  n_packed_vec      = %d (向量化后)\n", n_packed_vec);
    printf("  threads_per_block = %d\n", threads_per_block);
    printf("  grid_size         = %d\n", num_blocks_kernel);
    printf("  log2_blocksize    = %d\n", log2_bs);
    printf("  log2_s2_blocksize = %d\n", log2_s2);

    // 预热
    printf("\n[INFO] 预热 %d 次...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        if (use_bf16) {
            nf4_dequantize_kernel<__nv_bfloat16><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, (__nv_bfloat16*)d_output
            );
        } else {
            nf4_dequantize_kernel<half><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, (half*)d_output
            );
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时: CUDA Events，每次迭代间同步以隔离测量
    printf("[INFO] 计时 %d 次...\n", repeats);

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<float> times(repeats);

    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (use_bf16) {
            nf4_dequantize_kernel<__nv_bfloat16><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, (__nv_bfloat16*)d_output
            );
        } else {
            nf4_dequantize_kernel<half><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, (half*)d_output
            );
        }
        CUDA_CHECK(cudaEventRecord(ev_end));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], ev_start, ev_end));
    }

    // 排序取中位数，抗干扰
    std::vector<float> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    float total_ms = 0.0f;
    float min_ms = sorted_times.front();
    float max_ms = sorted_times.back();
    for (int i = 0; i < repeats; i++) total_ms += times[i];
    float avg_ms = total_ms / repeats;
    float median_ms = sorted_times[repeats / 2];

    // 有效内存带宽 (基于中位数)
    double read_bytes = (double)packed_size + data.num_blocks + data.num_groups * 2 + 256 * 2;
    double write_bytes = (double)output_bytes;
    double total_bytes = read_bytes + write_bytes;
    double bandwidth_gbps = total_bytes / (median_ms * 1e-3) / 1e9;

    printf("\n========================================\n");
    printf("  NF4 反量化 Kernel 性能\n");
    printf("========================================\n");
    printf("  矩阵大小     : (%ld, %ld)\n", (long)data.num_rows, (long)data.num_cols);
    printf("  块大小       : %d\n", data.blocksize);
    printf("  输出类型     : %s\n", compute_type.c_str());
    printf("  平均耗时     : %.4f ms\n", avg_ms);
    printf("  中位数耗时   : %.4f ms\n", median_ms);
    printf("  最小耗时     : %.4f ms\n", min_ms);
    printf("  最大耗时     : %.4f ms\n", max_ms);
    printf("  有效带宽     : %.2f GB/s (基于中位数)\n", bandwidth_gbps);
    printf("========================================\n");

    // 写出结果
    std::vector<uint8_t> h_output(output_bytes);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    FILE* fout = fopen(output_file, "wb");
    if (!fout) {
        fprintf(stderr, "[ERROR] Cannot open output file: %s\n", output_file);
        return 1;
    }
    fwrite(h_output.data(), 1, output_bytes, fout);
    fclose(fout);
    printf("\n[INFO] 已写入解量化输出: %s (%ld bytes)\n", output_file, (long)output_bytes);

    // 清理
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    CUDA_CHECK(cudaFree(d_packed_weights));
    CUDA_CHECK(cudaFree(d_absmax_q));
    CUDA_CHECK(cudaFree(d_absmax2));
    CUDA_CHECK(cudaFree(d_code2));
    CUDA_CHECK(cudaFree(d_output));

    printf("[DONE] 完成\n");
    return 0;
}
