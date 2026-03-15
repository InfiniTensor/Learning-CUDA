#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
#error "CUDA-compatible runtime header not found. Please install Iluvatar SDK and set include paths."
#endif

#include "nf4_dequant_kernel.cuh"

#define ILUVATAR_BACKEND_NAME "ILUVATAR (CUDA-Compatible)"

#define ILU_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "Runtime error at %s:%d: %s\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err__));     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct NF4Data {
    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int32_t blocksize = 0;

    std::vector<uint8_t> packed_weights;
    std::vector<uint8_t> absmax_q;
    std::vector<uint16_t> absmax2;
    std::vector<uint16_t> code2;
    float offset = 0.0f;

    int64_t n_elements = 0;
    int32_t num_blocks = 0;
    int32_t num_groups = 0;
    int32_t s2_blocksize = 0;
};

static bool is_power_of_two(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

static bool read_nf4_data(const char* filepath, NF4Data& data) {
    FILE* f = std::fopen(filepath, "rb");
    if (!f) {
        std::fprintf(stderr, "[ERROR] Cannot open file: %s\n", filepath);
        return false;
    }

    if (std::fread(&data.num_rows, sizeof(int64_t), 1, f) != 1 ||
        std::fread(&data.num_cols, sizeof(int64_t), 1, f) != 1 ||
        std::fread(&data.blocksize, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Bad header in file: %s\n", filepath);
        return false;
    }

    data.n_elements = data.num_rows * data.num_cols;
    data.num_blocks = (int32_t)((data.n_elements + data.blocksize - 1) / data.blocksize);

    int64_t packed_size = data.n_elements / 2;
    data.packed_weights.resize(packed_size);
    if (std::fread(data.packed_weights.data(), 1, packed_size, f) != (size_t)packed_size) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Bad packed data in file: %s\n", filepath);
        return false;
    }

    data.absmax_q.resize(data.num_blocks);
    if (std::fread(data.absmax_q.data(), 1, data.num_blocks, f) != (size_t)data.num_blocks) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Bad absmax_q in file: %s\n", filepath);
        return false;
    }

    long current_pos = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, current_pos, SEEK_SET);

    long remaining = file_size - current_pos;
    long fixed_tail = 256 * 2 + 4;
    long absmax2_bytes = remaining - fixed_tail;

    if (absmax2_bytes <= 0 || (absmax2_bytes % 2) != 0) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Invalid absmax2 segment in file: %s\n", filepath);
        return false;
    }

    data.num_groups = (int32_t)(absmax2_bytes / 2);
    data.s2_blocksize = (data.num_blocks + data.num_groups - 1) / data.num_groups;

    data.absmax2.resize(data.num_groups);
    if (std::fread(data.absmax2.data(), 2, data.num_groups, f) != (size_t)data.num_groups) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Bad absmax2 in file: %s\n", filepath);
        return false;
    }

    data.code2.resize(256);
    if (std::fread(data.code2.data(), 2, 256, f) != 256) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Bad code2 in file: %s\n", filepath);
        return false;
    }

    if (std::fread(&data.offset, sizeof(float), 1, f) != 1) {
        std::fclose(f);
        std::fprintf(stderr, "[ERROR] Missing offset in file: %s\n", filepath);
        return false;
    }

    std::fclose(f);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]\n", argv[0]);
        return 1;
    }

    const char* weight_file = argv[1];
    const char* output_file = argv[2];
    std::string compute_type = (argc > 3) ? argv[3] : "bf16";
    int warmup = (argc > 4) ? std::atoi(argv[4]) : 10;
    int repeats = (argc > 5) ? std::atoi(argv[5]) : 100;

    bool use_bf16 = (compute_type == "bf16");
    if (!use_bf16 && compute_type != "fp16") {
        std::fprintf(stderr, "[ERROR] compute_type must be bf16 or fp16\n");
        return 1;
    }

    std::printf("[INFO] Backend: %s\n", ILUVATAR_BACKEND_NAME);
    std::printf("[INFO] Loading weight file: %s\n", weight_file);

    NF4Data data;
    if (!read_nf4_data(weight_file, data)) {
        return 1;
    }

    std::printf("  num_rows     = %ld\n", (long)data.num_rows);
    std::printf("  num_cols     = %ld\n", (long)data.num_cols);
    std::printf("  blocksize    = %d\n", data.blocksize);
    std::printf("  n_elements   = %ld\n", (long)data.n_elements);
    std::printf("  num_blocks   = %d\n", data.num_blocks);
    std::printf("  num_groups   = %d\n", data.num_groups);
    std::printf("  s2_blocksize = %d\n", data.s2_blocksize);
    std::printf("  offset       = %f\n", data.offset);
    std::printf("  compute_type = %s\n", compute_type.c_str());

    if (!is_power_of_two(data.blocksize) || !is_power_of_two(data.s2_blocksize)) {
        std::fprintf(stderr,
                     "[ERROR] blocksize and s2_blocksize must be powers of two. got blocksize=%d s2_blocksize=%d\n",
                     data.blocksize, data.s2_blocksize);
        return 1;
    }

    uint8_t* d_packed_weights = nullptr;
    uint8_t* d_absmax_q = nullptr;
    uint16_t* d_absmax2 = nullptr;
    uint16_t* d_code2 = nullptr;
    uint16_t* d_output_bits = nullptr;

    int64_t packed_size = data.n_elements / 2;
    int64_t output_bytes = data.n_elements * 2;

    ILU_CHECK(cudaMalloc((void**)&d_packed_weights, packed_size));
    ILU_CHECK(cudaMalloc((void**)&d_absmax_q, data.num_blocks));
    ILU_CHECK(cudaMalloc((void**)&d_absmax2, data.num_groups * sizeof(uint16_t)));
    ILU_CHECK(cudaMalloc((void**)&d_code2, 256 * sizeof(uint16_t)));
    ILU_CHECK(cudaMalloc((void**)&d_output_bits, output_bytes));

    ILU_CHECK(cudaMemcpy(d_packed_weights, data.packed_weights.data(),
                         packed_size, cudaMemcpyHostToDevice));
    ILU_CHECK(cudaMemcpy(d_absmax_q, data.absmax_q.data(),
                         data.num_blocks, cudaMemcpyHostToDevice));
    ILU_CHECK(cudaMemcpy(d_absmax2, data.absmax2.data(),
                         data.num_groups * sizeof(uint16_t), cudaMemcpyHostToDevice));
    ILU_CHECK(cudaMemcpy(d_code2, data.code2.data(),
                         256 * sizeof(uint16_t), cudaMemcpyHostToDevice));

    int n_packed = (int)((data.n_elements + 1) / 2);
    int n_packed_vec = (n_packed + 3) / 4;
    int threads_per_block = 256;
    int num_blocks_kernel = (n_packed_vec + threads_per_block - 1) / threads_per_block;
    int log2_bs = log2_pow2(data.blocksize);
    int log2_s2 = log2_pow2(data.s2_blocksize);

    std::printf("\n[INFO] Kernel config:\n");
    std::printf("  n_packed          = %d\n", n_packed);
    std::printf("  n_packed_vec      = %d\n", n_packed_vec);
    std::printf("  threads_per_block = %d\n", threads_per_block);
    std::printf("  grid_size         = %d\n", num_blocks_kernel);
    std::printf("  log2_blocksize    = %d\n", log2_bs);
    std::printf("  log2_s2_blocksize = %d\n", log2_s2);

    std::printf("\n[INFO] Warmup %d iterations...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        if (use_bf16) {
            nf4_dequantize_kernel<true><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, d_output_bits);
        } else {
            nf4_dequantize_kernel<false><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, d_output_bits);
        }
        ILU_CHECK(cudaGetLastError());
    }
    ILU_CHECK(cudaDeviceSynchronize());

    std::printf("[INFO] Timing %d iterations...\n", repeats);

    cudaEvent_t ev_start;
    cudaEvent_t ev_end;
    ILU_CHECK(cudaEventCreate(&ev_start));
    ILU_CHECK(cudaEventCreate(&ev_end));

    std::vector<float> times(repeats);

    for (int i = 0; i < repeats; ++i) {
        ILU_CHECK(cudaDeviceSynchronize());
        ILU_CHECK(cudaEventRecord(ev_start));

        if (use_bf16) {
            nf4_dequantize_kernel<true><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, d_output_bits);
        } else {
            nf4_dequantize_kernel<false><<<num_blocks_kernel, threads_per_block>>>(
                d_packed_weights, d_absmax_q, d_absmax2, d_code2,
                data.offset, log2_bs, log2_s2,
                data.n_elements, d_output_bits);
        }

        ILU_CHECK(cudaGetLastError());
        ILU_CHECK(cudaEventRecord(ev_end));
        ILU_CHECK(cudaEventSynchronize(ev_end));
        ILU_CHECK(cudaEventElapsedTime(&times[i], ev_start, ev_end));
    }

    std::vector<float> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    float total_ms = 0.0f;
    for (float t : times) {
        total_ms += t;
    }

    float min_ms = sorted_times.front();
    float max_ms = sorted_times.back();
    float avg_ms = total_ms / repeats;
    float median_ms = sorted_times[repeats / 2];

    double read_bytes = (double)packed_size + data.num_blocks +
                        data.num_groups * 2.0 + 256.0 * 2.0;
    double write_bytes = (double)output_bytes;
    double total_bytes = read_bytes + write_bytes;
    double bandwidth_gbps = total_bytes / (median_ms * 1e-3) / 1e9;

    std::printf("\n========================================\n");
    std::printf("  NF4 Dequant Kernel Performance (ILUVATAR)\n");
    std::printf("========================================\n");
    std::printf("  matrix shape   : (%ld, %ld)\n", (long)data.num_rows, (long)data.num_cols);
    std::printf("  blocksize      : %d\n", data.blocksize);
    std::printf("  output type    : %s\n", compute_type.c_str());
    std::printf("  avg latency    : %.4f ms\n", avg_ms);
    std::printf("  median latency : %.4f ms\n", median_ms);
    std::printf("  min latency    : %.4f ms\n", min_ms);
    std::printf("  max latency    : %.4f ms\n", max_ms);
    std::printf("  bandwidth      : %.2f GB/s (median)\n", bandwidth_gbps);
    std::printf("========================================\n");

    std::vector<uint16_t> h_output_bits(data.n_elements);
    ILU_CHECK(cudaMemcpy(h_output_bits.data(), d_output_bits, output_bytes, cudaMemcpyDeviceToHost));

    FILE* fout = std::fopen(output_file, "wb");
    if (!fout) {
        std::fprintf(stderr, "[ERROR] Cannot open output file: %s\n", output_file);
        return 1;
    }

    std::fwrite(h_output_bits.data(), sizeof(uint16_t), h_output_bits.size(), fout);
    std::fclose(fout);

    std::printf("\n[INFO] Wrote output: %s (%ld bytes)\n", output_file, (long)output_bytes);

    ILU_CHECK(cudaEventDestroy(ev_start));
    ILU_CHECK(cudaEventDestroy(ev_end));
    ILU_CHECK(cudaFree(d_packed_weights));
    ILU_CHECK(cudaFree(d_absmax_q));
    ILU_CHECK(cudaFree(d_absmax2));
    ILU_CHECK(cudaFree(d_code2));
    ILU_CHECK(cudaFree(d_output_bits));

    std::printf("[DONE] Finished\n");
    return 0;
}
