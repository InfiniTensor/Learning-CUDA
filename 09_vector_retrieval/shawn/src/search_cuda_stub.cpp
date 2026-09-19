#include "vector_search/search.hpp"

#include <stdexcept>

namespace vector_search {

bool cuda_backend_available() { return false; }

SearchResults search_exact_cuda(const VectorDatabase&, const QuerySet&, int,
                                int) {
  throw std::runtime_error(
      "CUDA backend was not built; configure on a machine with nvcc");
}

SearchResults search_candidates_cuda(
    const VectorDatabase&, const QuerySet&,
    const std::vector<std::vector<std::int64_t>>&, int, int) {
  throw std::runtime_error(
      "CUDA backend was not built; configure on a machine with nvcc");
}

std::vector<int> assign_nearest_cuda(const std::vector<float>&, std::int64_t,
                                     int, const std::vector<float>&, int,
                                     Metric) {
  throw std::runtime_error(
      "CUDA backend was not built; configure on a machine with nvcc");
}

}  // namespace vector_search
