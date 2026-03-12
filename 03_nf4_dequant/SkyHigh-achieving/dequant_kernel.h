#pragma once

#include <cstdint>
#include <vector>

enum class ComputeType {
    FP16,
    BF16
};

struct DequantConfig {
    int64_t rows;
    int64_t cols;
    int32_t blocksize;
    ComputeType compute_type;
};

struct NF4Binary {
    DequantConfig config;
    std::vector<uint8_t> packed_weights;
    std::vector<uint8_t> absmax_q;
    std::vector<uint16_t> absmax2_raw;
    std::vector<uint16_t> code2_raw;
    float offset;
};

bool load_nf4_binary(const char* file_path, NF4Binary& out);
bool save_float_output(const char* file_path, const std::vector<float>& data);
bool run_dequant_cuda(const NF4Binary& input, std::vector<float>& output, float& mae);
