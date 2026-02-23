#ifndef NF4_DEQUANT_H_
#define NF4_DEQUANT_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

struct NF4QuantState {
    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int32_t blocksize = 0;
    int32_t blocks_per_group = 256;

    uint8_t* h_packed_weights = nullptr;
    uint8_t* h_absmax_q = nullptr;
    __half* h_absmax2 = nullptr;
    __half* h_code2 = nullptr;
    float h_offset = 0.0f;

    size_t num_elements = 0;
    size_t num_packed_bytes = 0;
    size_t num_blocks = 0;
    size_t num_groups = 0;
};

bool load_nf4_file(const char* bin_path, NF4QuantState* state);
void free_nf4_state(NF4QuantState* state);
void save_dequant(const void* data, int64_t rows, int64_t cols, const char* out_path, bool is_bf16);
void cpu_dequant_nf4(const NF4QuantState& state, void* output, bool use_bf16);
bool cuda_dequant_nf4(
    const NF4QuantState& state,
    void* output,
    bool use_bf16,
    float* kernel_time_ms,
    int block_dim = 256,
    bool copy_output_to_host = true,
    bool reuse_device_buffers = true);
void cuda_release_nf4_device_cache();

#endif  // NF4_DEQUANT_H_
