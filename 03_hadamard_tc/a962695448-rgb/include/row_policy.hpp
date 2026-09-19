#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hadamard {

enum class RowLayout { Original, Packed, Auto };
enum class RowDevice { Unknown, A800, RTX4090, A100, RTX4090D };
struct RowChoice { bool packed; int threads; };

inline RowLayout parse_row_layout(const std::string& name) {
    if (name == "original") return RowLayout::Original;
    if (name == "packed") return RowLayout::Packed;
    if (name == "auto") return RowLayout::Auto;
    throw std::invalid_argument("row_layout must be original, packed, or auto");
}

inline RowDevice row_device(const char* name) {
    if (std::strcmp(name, "NVIDIA A800-SXM4-40GB") == 0) return RowDevice::A800;
    if (std::strcmp(name, "NVIDIA GeForce RTX 4090") == 0) return RowDevice::RTX4090;
    if (std::strcmp(name, "NVIDIA A100-SXM4-40GB") == 0) return RowDevice::A100;
    if (std::strcmp(name, "NVIDIA GeForce RTX 4090 D") == 0) return RowDevice::RTX4090D;
    return RowDevice::Unknown;
}

inline RowChoice choose_rows(RowLayout layout, RowDevice device, std::size_t rows,
                             int dim, bool fused, int fallback_threads) {
    if (layout == RowLayout::Original || dim < 1 || dim > 16 || (dim & (dim - 1)))
        return {false, fallback_threads};
    if (layout == RowLayout::Packed) return {true, fallback_threads};
    // This model has separate measured bounds; do not inherit the 4090 rules.
    if (device == RowDevice::RTX4090D)
        return rows >= 4096 && rows <= 65536 ? RowChoice{true, 256}
                                           : RowChoice{false, fallback_threads};
    int index = 0;
    for (int n = dim; n > 1; n /= 2) ++index;
    // Zero disables a rule. Only independently accepted device-specific ranges are installed.
    // BEGIN GENERATED RULES
    constexpr std::size_t a800_transform[5] = {4096, 4096, 4096, 4096, 4096};
    constexpr std::size_t a800_fused[5] = {1, 1, 16, 256, 0};
    constexpr std::size_t rtx4090_transform[5] = {4096, 4096, 4096, 4096, 4096};
    constexpr std::size_t rtx4090_fused[5] = {1, 1, 256, 64, 0};
    constexpr std::size_t a100_transform[5] = {4096, 256, 4096, 4096, 4096};
    constexpr std::size_t a100_fused[5] = {1, 1, 16, 64, 0};
    // END GENERATED RULES
    std::size_t minimum = 0;
    if (device == RowDevice::A800) minimum = (fused ? a800_fused : a800_transform)[index];
    if (device == RowDevice::RTX4090) minimum = (fused ? rtx4090_fused : rtx4090_transform)[index];
    if (device == RowDevice::A100) minimum = (fused ? a100_fused : a100_transform)[index];
    return minimum && rows >= minimum ? RowChoice{true, 256} : RowChoice{false, fallback_threads};
}

}  // namespace hadamard
