#include "gvs/search.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gvs {
namespace {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(error__));               \
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count = 0) : count_(count) {
        if (count_) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void distance_kernel(const float* database, const float* queries,
                                float* distances, std::size_t n,
                                std::size_t batch, std::size_t dim,
                                int metric) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t q = blockIdx.y;
    if (i >= n || q >= batch) return;
    const float* x = database + i * dim;
    const float* query = queries + q * dim;
    float value = 0.0f;
    if (metric == 0) {
        for (std::size_t d = 0; d < dim; ++d) {
            const float diff = query[d] - x[d];
            value = fmaf(diff, diff, value);
        }
    } else {
        for (std::size_t d = 0; d < dim; ++d)
            value = fmaf(query[d], x[d], value);
        value = -value;
    }
    distances[q * n + i] = value;
}

__global__ void indexed_distance_kernel(const float* database, const float* query,
                                        const int* ids, float* distances,
                                        std::size_t candidates, std::size_t dim,
                                        int metric) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidates) return;
    const float* x = database + static_cast<std::size_t>(ids[i]) * dim;
    float value = 0.0f;
    if (metric == 0) {
        for (std::size_t d = 0; d < dim; ++d) {
            const float diff = query[d] - x[d];
            value = fmaf(diff, diff, value);
        }
    } else {
        for (std::size_t d = 0; d < dim; ++d)
            value = fmaf(query[d], x[d], value);
        value = -value;
    }
    distances[i] = value;
}

SearchResult finalize_on_cpu(const std::vector<float>& distances,
                             std::size_t nq, std::size_t n,
                             std::size_t k, Metric metric) {
    SearchResult output{nq, k, std::vector<int>(nq * k),
                        std::vector<float>(nq * k)};
    std::vector<std::pair<float, int>> row(n);
    for (std::size_t q = 0; q < nq; ++q) {
        for (std::size_t i = 0; i < n; ++i)
            row[i] = {distances[q * n + i], static_cast<int>(i)};
        std::partial_sort(row.begin(), row.begin() + k, row.end());
        for (std::size_t j = 0; j < k; ++j) {
            output.ids[q * k + j] = row[j].second;
            output.scores[q * k + j] =
                metric == Metric::InnerProduct ? -row[j].first : row[j].first;
        }
    }
    return output;
}

}  // namespace

SearchResult gpu_exact_search(const std::vector<float>& database,
                              const std::vector<float>& queries,
                              std::size_t n, std::size_t nq,
                              std::size_t dim, Metric metric,
                              const SearchConfig& config,
                              GpuStats* stats) {
    if (database.size() != n * dim || queries.size() != nq * dim)
        throw std::invalid_argument("input shape does not match n/nq/dim");
    const std::size_t k = std::min(config.k, n);
    const std::size_t batch_capacity = std::min(config.batch_size, nq);

    DeviceBuffer<float> d_database(database.size());
    DeviceBuffer<float> d_queries(batch_capacity * dim);
    DeviceBuffer<float> d_distances(batch_capacity * n);
    CUDA_CHECK(cudaMemcpy(d_database.get(), database.data(),
                          database.size() * sizeof(float), cudaMemcpyHostToDevice));

    SearchResult output{nq, k, std::vector<int>(nq * k),
                        std::vector<float>(nq * k)};
    std::vector<float> host_distances(batch_capacity * n);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto begin = std::chrono::steady_clock::now();

    for (std::size_t first = 0; first < nq; first += batch_capacity) {
        const std::size_t batch = std::min(batch_capacity, nq - first);
        CUDA_CHECK(cudaMemcpy(d_queries.get(), queries.data() + first * dim,
                              batch * dim * sizeof(float), cudaMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                        static_cast<unsigned>(batch));
        distance_kernel<<<grid, block>>>(d_database.get(), d_queries.get(),
                                         d_distances.get(), n, batch, dim,
                                         metric == Metric::L2 ? 0 : 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_distances.data(), d_distances.get(),
                              batch * n * sizeof(float), cudaMemcpyDeviceToHost));
        SearchResult part = finalize_on_cpu(host_distances, batch, n, k, metric);
        std::copy(part.ids.begin(), part.ids.end(), output.ids.begin() + first * k);
        std::copy(part.scores.begin(), part.scores.end(),
                  output.scores.begin() + first * k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(end - begin).count();

    if (stats) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        stats->elapsed_ms = elapsed;
        stats->peak_workspace_bytes =
            (database.size() + batch_capacity * dim + batch_capacity * n) *
            sizeof(float);
        stats->device_name = prop.name;
    }
    return output;
}

SearchResult gpu_indexed_search(const std::vector<float>& database,
                                const std::vector<float>& queries,
                                const std::vector<std::vector<int>>& candidate_ids,
                                std::size_t n, std::size_t nq,
                                std::size_t dim, Metric metric,
                                const SearchConfig& config,
                                GpuStats* stats) {
    if (database.size() != n * dim || queries.size() != nq * dim ||
        candidate_ids.size() != nq)
        throw std::invalid_argument("indexed search shape mismatch");
    const std::size_t k = std::min(config.k, n);
    std::size_t max_candidates = 0;
    for (const auto& ids : candidate_ids) max_candidates = std::max(max_candidates, ids.size());
    SearchResult output{nq, k, std::vector<int>(nq * k, -1),
                        std::vector<float>(nq * k, 0.0f)};
    if (max_candidates == 0) return output;

    DeviceBuffer<float> d_database(database.size());
    DeviceBuffer<float> d_query(dim);
    DeviceBuffer<int> d_ids(max_candidates);
    DeviceBuffer<float> d_distances(max_candidates);
    CUDA_CHECK(cudaMemcpy(d_database.get(), database.data(),
                          database.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> host_distances(max_candidates);
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < nq; ++q) {
        const std::size_t count = candidate_ids[q].size();
        if (count == 0) continue;
        CUDA_CHECK(cudaMemcpy(d_query.get(), queries.data() + q * dim,
                              dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids.get(), candidate_ids[q].data(),
                              count * sizeof(int), cudaMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((count + block.x - 1) / block.x));
        indexed_distance_kernel<<<grid, block>>>(
            d_database.get(), d_query.get(), d_ids.get(), d_distances.get(),
            count, dim, metric == Metric::L2 ? 0 : 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_distances.data(), d_distances.get(),
                              count * sizeof(float), cudaMemcpyDeviceToHost));
        const std::size_t part_k = std::min(k, count);
        std::vector<std::pair<float, int>> row(count);
        for (std::size_t i = 0; i < count; ++i)
            row[i] = {host_distances[i], candidate_ids[q][i]};
        std::partial_sort(row.begin(), row.begin() + part_k, row.end());
        for (std::size_t j = 0; j < part_k; ++j) {
            output.ids[q * k + j] = row[j].second;
            output.scores[q * k + j] =
                metric == Metric::InnerProduct ? -row[j].first : row[j].first;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (stats) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        stats->elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        stats->peak_workspace_bytes = database.size() * sizeof(float) +
            dim * sizeof(float) + max_candidates * (sizeof(float) + sizeof(int));
        stats->device_name = prop.name;
    }
    return output;
}

}  // namespace gvs
