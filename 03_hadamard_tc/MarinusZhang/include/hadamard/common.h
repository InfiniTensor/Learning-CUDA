// Common types and constants shared by the CPU reference and the CUDA kernels.
#ifndef HADAMARD_COMMON_H_
#define HADAMARD_COMMON_H_

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace hadamard {

// Logical view of the input tensor [batch, seq_len, num_heads, head_dim].
//
// The transform runs along the contiguous last dimension, so the tensor is
// equivalent to rows() independent vectors of length head_dim. head_dim must be a
// power of two; non power-of-two sizes are out of scope for this project.
struct Shape {
    int batch = 1;
    int seq_len = 1;
    int num_heads = 1;
    int head_dim = 128;

    long rows() const { return static_cast<long>(batch) * seq_len * num_heads; }
    long elems() const { return rows() * head_dim; }
    bool valid() const;

    std::string to_string() const {
        return "[" + std::to_string(batch) + "," + std::to_string(seq_len) + "," +
               std::to_string(num_heads) + "," + std::to_string(head_dim) + "]";
    }
};

enum class DType { kFp16, kBFloat16 };

inline const char* dtype_name(DType dtype) {
    return dtype == DType::kFp16 ? "fp16" : "bf16";
}

// Correctness thresholds from the project spec (absolute error).
constexpr float kAtolFp16 = 1e-2f;
constexpr float kAtolBf16 = 5e-2f;

// FP8 E4M3 (S1E4M3, exponent bias 7) tops out at 448. A per-token scale maps the
// largest magnitude of that row onto it: scale = amax / kFp8E4M3Max, and
// dequantisation is value = code_value * scale.
constexpr float kFp8E4M3Max = 448.0f;

inline float tolerance_for(DType dtype) {
    return dtype == DType::kFp16 ? kAtolFp16 : kAtolBf16;
}

inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline int log2_int(int n) {
    int k = 0;
    while ((1 << k) < n) {
        ++k;
    }
    return k;
}

inline bool Shape::valid() const {
    return batch > 0 && seq_len > 0 && num_heads > 0 && is_power_of_two(head_dim) && head_dim >= 2;
}

// Deterministic host-side data in [-1, 1): the same seed gives the same input for
// every implementation under test.
inline std::vector<float> random_host_data(long n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n));
    for (long i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = dist(gen);
    }
    return data;
}

}  // namespace hadamard

#endif  // HADAMARD_COMMON_H_
