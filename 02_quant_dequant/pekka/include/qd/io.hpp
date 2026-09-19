#pragma once

#include "qd/common.hpp"

#include <filesystem>
#include <vector>

namespace qd {

Matrix read_qdat(const std::filesystem::path &path);
void write_qdat(const std::filesystem::path &path, std::uint64_t rows,
                std::uint64_t cols, DataType type,
                const std::vector<std::uint8_t> &payload);

void write_qlow(const std::filesystem::path &path, const QuantizedData &data);
QuantizedData read_qlow(const std::filesystem::path &path);
std::uint64_t qlow_file_size(const QuantizedData &data);

} // namespace qd
