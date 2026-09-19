#pragma once

#include "qd/common.hpp"

#include <filesystem>

namespace qd {

Config read_config(const std::filesystem::path &path);
void validate_config(const Config &config);

} // namespace qd
