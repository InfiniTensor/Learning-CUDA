#pragma once

#include "gvs/search.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gvs {

struct VectorFile {
    std::size_t count = 0;
    std::size_t dim = 0;
    Metric metric = Metric::L2;
    std::vector<float> values;
};

struct Parameters {
    std::size_t top_k = 10;
    std::string search_mode = "exact";
    std::size_t batch_size = 128;
    std::size_t nlist = 4096;
    std::size_t nprobe = 16;
};

void write_vector_file(const std::string& path, const VectorFile& data,
                       bool include_metric);
VectorFile read_vector_file(const std::string& path, bool expect_metric);
void write_search_result(const std::string& path, const SearchResult& result);
Parameters read_parameters(const std::string& path);

}  // namespace gvs

