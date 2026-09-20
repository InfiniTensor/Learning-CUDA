#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gvs {

enum class Metric { L2, InnerProduct };

struct SearchConfig {
    std::size_t k = 10;
    std::size_t batch_size = 16;
};

struct SearchResult {
    std::size_t queries = 0;
    std::size_t k = 0;
    std::vector<int> ids;
    std::vector<float> scores;
};

struct GpuStats {
    double elapsed_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
    std::string device_name;
};

SearchResult cpu_exact_search(const std::vector<float>& database,
                              const std::vector<float>& queries,
                              std::size_t n, std::size_t nq,
                              std::size_t dim, Metric metric,
                              std::size_t k);

SearchResult gpu_exact_search(const std::vector<float>& database,
                              const std::vector<float>& queries,
                              std::size_t n, std::size_t nq,
                              std::size_t dim, Metric metric,
                              const SearchConfig& config,
                              GpuStats* stats = nullptr);

SearchResult gpu_indexed_search(const std::vector<float>& database,
                                const std::vector<float>& queries,
                                const std::vector<std::vector<int>>& candidate_ids,
                                std::size_t n, std::size_t nq,
                                std::size_t dim, Metric metric,
                                const SearchConfig& config,
                                GpuStats* stats = nullptr);

double recall_at_k(const SearchResult& truth, const SearchResult& candidate);

class IvfFlatIndex {
public:
    IvfFlatIndex(std::size_t dim, std::size_t nlist, Metric metric);

    void train(const std::vector<float>& database, std::size_t n,
               std::size_t iterations = 12, std::uint64_t seed = 42);
    void add(const std::vector<float>& database, std::size_t n);
    SearchResult search(const std::vector<float>& queries, std::size_t nq,
                        std::size_t nprobe, const SearchConfig& config,
                        GpuStats* stats = nullptr) const;

    void save(const std::string& path) const;
    static IvfFlatIndex load(const std::string& path);

    std::size_t nlist() const { return nlist_; }

private:
    std::size_t dim_;
    std::size_t nlist_;
    Metric metric_;
    std::vector<float> centroids_;
    std::vector<float> database_;
    std::vector<std::vector<int>> lists_;
};

}  // namespace gvs
