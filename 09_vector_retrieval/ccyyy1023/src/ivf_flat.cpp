#include "gvs/search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace gvs {
namespace {

float coarse_distance(const float* a, const float* b, std::size_t dim,
                      Metric metric) {
    float value = 0.0f;
    if (metric == Metric::L2) {
        for (std::size_t d = 0; d < dim; ++d) {
            const float diff = a[d] - b[d];
            value += diff * diff;
        }
        return value;
    }
    for (std::size_t d = 0; d < dim; ++d) value += a[d] * b[d];
    return -value;
}

std::size_t nearest_centroid(const float* x, const std::vector<float>& centroids,
                             std::size_t nlist, std::size_t dim, Metric metric) {
    std::size_t best = 0;
    float best_value = std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < nlist; ++c) {
        const float value = coarse_distance(x, centroids.data() + c * dim,
                                            dim, metric);
        if (value < best_value) { best_value = value; best = c; }
    }
    return best;
}

}  // namespace

namespace {
constexpr char kIvfMagic[8] = {'G','V','S','I','V','F','1','\0'};
template <typename T>
void write_value(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed to write IVF index");
}
template <typename T>
T read_value(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated IVF index");
    return value;
}
}

IvfFlatIndex::IvfFlatIndex(std::size_t dim, std::size_t nlist, Metric metric)
    : dim_(dim), nlist_(nlist), metric_(metric), lists_(nlist) {
    if (dim == 0 || nlist == 0) throw std::invalid_argument("dim/nlist must be positive");
}

void IvfFlatIndex::train(const std::vector<float>& database, std::size_t n,
                         std::size_t iterations, std::uint64_t seed) {
    if (database.size() != n * dim_ || n < nlist_)
        throw std::invalid_argument("invalid training matrix or n < nlist");
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    centroids_.resize(nlist_ * dim_);
    for (std::size_t c = 0; c < nlist_; ++c)
        std::copy_n(database.data() + order[c] * dim_, dim_,
                    centroids_.data() + c * dim_);

    std::vector<float> sums(nlist_ * dim_);
    std::vector<std::size_t> counts(nlist_);
    // Mini-batch Lloyd training bounds build cost on million-vector inputs.
    const std::size_t train_count = std::min(n, std::max<std::size_t>(4096, nlist_ * 4));
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t sample = 0; sample < train_count; ++sample) {
            const std::size_t i = order[sample];
            const std::size_t c = nearest_centroid(database.data() + i * dim_,
                                                   centroids_, nlist_, dim_, metric_);
            ++counts[c];
            for (std::size_t d = 0; d < dim_; ++d)
                sums[c * dim_ + d] += database[i * dim_ + d];
        }
        for (std::size_t c = 0; c < nlist_; ++c) {
            if (counts[c] == 0) continue;
            for (std::size_t d = 0; d < dim_; ++d)
                centroids_[c * dim_ + d] = sums[c * dim_ + d] / counts[c];
        }
    }
}

void IvfFlatIndex::add(const std::vector<float>& database, std::size_t n) {
    if (centroids_.empty()) throw std::logic_error("train must be called before add");
    if (database.size() != n * dim_) throw std::invalid_argument("invalid database shape");
    database_ = database;
    lists_.assign(nlist_, {});
    std::vector<std::size_t> labels(n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        labels[i] = nearest_centroid(database.data() + i * dim_, centroids_,
                                     nlist_, dim_, metric_);
    }
    for (std::size_t i = 0; i < n; ++i)
        lists_[labels[i]].push_back(static_cast<int>(i));
}

SearchResult IvfFlatIndex::search(const std::vector<float>& queries,
                                  std::size_t nq, std::size_t nprobe,
                                  const SearchConfig& config,
                                  GpuStats* stats) const {
    if (database_.empty()) throw std::logic_error("add must be called before search");
    if (queries.size() != nq * dim_) throw std::invalid_argument("invalid query shape");
    nprobe = std::min(nprobe, nlist_);
    const std::size_t k = std::min(config.k, database_.size() / dim_);
    const auto search_begin = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> candidate_ids(nq);

    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, int>> coarse(nlist_);
        for (std::size_t c = 0; c < nlist_; ++c)
            coarse[c] = {coarse_distance(queries.data() + q * dim_,
                                         centroids_.data() + c * dim_, dim_, metric_),
                         static_cast<int>(c)};
        std::partial_sort(coarse.begin(), coarse.begin() + nprobe, coarse.end());
        for (std::size_t p = 0; p < nprobe; ++p) {
            const auto& list = lists_[coarse[p].second];
            candidate_ids[q].insert(candidate_ids[q].end(), list.begin(), list.end());
        }
    }
    GpuStats gpu_stats;
    SearchResult output = gpu_indexed_search(
        database_, queries, candidate_ids, database_.size() / dim_, nq, dim_,
        metric_, SearchConfig{k, 1}, &gpu_stats);
    if (stats) {
        stats->elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - search_begin).count();
        stats->peak_workspace_bytes = gpu_stats.peak_workspace_bytes;
        stats->device_name = gpu_stats.device_name;
    }
    return output;
}

void IvfFlatIndex::save(const std::string& path) const {
    if (database_.empty() || centroids_.empty())
        throw std::logic_error("cannot save an unbuilt IVF index");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open IVF index output: " + path);
    out.write(kIvfMagic, 8);
    write_value(out, static_cast<std::uint64_t>(database_.size() / dim_));
    write_value(out, static_cast<std::uint32_t>(dim_));
    write_value(out, static_cast<std::uint32_t>(nlist_));
    write_value(out, static_cast<std::uint32_t>(metric_ == Metric::L2 ? 0 : 1));
    out.write(reinterpret_cast<const char*>(centroids_.data()),
              static_cast<std::streamsize>(centroids_.size() * sizeof(float)));
    out.write(reinterpret_cast<const char*>(database_.data()),
              static_cast<std::streamsize>(database_.size() * sizeof(float)));
    for (const auto& list : lists_) {
        write_value(out, static_cast<std::uint64_t>(list.size()));
        out.write(reinterpret_cast<const char*>(list.data()),
                  static_cast<std::streamsize>(list.size() * sizeof(int)));
    }
    if (!out) throw std::runtime_error("failed to write IVF payload");
}

IvfFlatIndex IvfFlatIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open IVF index: " + path);
    char magic[8]{};
    in.read(magic, 8);
    if (!in || std::memcmp(magic, kIvfMagic, 8) != 0)
        throw std::runtime_error("invalid IVF index magic/version");
    const std::size_t n = static_cast<std::size_t>(read_value<std::uint64_t>(in));
    const std::size_t dim = read_value<std::uint32_t>(in);
    const std::size_t nlist = read_value<std::uint32_t>(in);
    const auto metric_code = read_value<std::uint32_t>(in);
    if (n == 0 || dim == 0 || nlist == 0 || metric_code > 1 ||
        n > static_cast<std::size_t>(-1) / dim)
        throw std::runtime_error("invalid IVF index header");
    IvfFlatIndex index(dim, nlist,
                       metric_code == 0 ? Metric::L2 : Metric::InnerProduct);
    index.centroids_.resize(nlist * dim);
    index.database_.resize(n * dim);
    in.read(reinterpret_cast<char*>(index.centroids_.data()),
            static_cast<std::streamsize>(index.centroids_.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(index.database_.data()),
            static_cast<std::streamsize>(index.database_.size() * sizeof(float)));
    std::size_t id_count = 0;
    for (auto& list : index.lists_) {
        const auto size = static_cast<std::size_t>(read_value<std::uint64_t>(in));
        if (size > n - id_count) throw std::runtime_error("invalid IVF list sizes");
        list.resize(size);
        in.read(reinterpret_cast<char*>(list.data()),
                static_cast<std::streamsize>(size * sizeof(int)));
        id_count += size;
        for (int id : list)
            if (id < 0 || static_cast<std::size_t>(id) >= n)
                throw std::runtime_error("invalid vector id in IVF index");
    }
    if (!in || id_count != n) throw std::runtime_error("truncated or incomplete IVF index");
    return index;
}

}  // namespace gvs
