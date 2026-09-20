#include "gvs/search.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_metric(gvs::Metric metric) {
    constexpr std::size_t n = 257;
    constexpr std::size_t nq = 7;
    constexpr std::size_t dim = 33;
    constexpr std::size_t k = 10;
    std::vector<float> database(n * dim);
    std::vector<float> queries(nq * dim);
    for (std::size_t i = 0; i < database.size(); ++i)
        database[i] = std::sin(static_cast<float>(i) * 0.013f);
    for (std::size_t i = 0; i < queries.size(); ++i)
        queries[i] = std::cos(static_cast<float>(i) * 0.017f);
    const auto cpu = gvs::cpu_exact_search(database, queries, n, nq, dim, metric, k);
    const auto gpu = gvs::gpu_exact_search(database, queries, n, nq, dim, metric,
                                           gvs::SearchConfig{k, 4});
    require(gvs::recall_at_k(cpu, gpu) == 1.0, "GPU exact Top-K differs from CPU");
    for (std::size_t i = 0; i < cpu.scores.size(); ++i)
        require(std::abs(cpu.scores[i] - gpu.scores[i]) < 2e-3f,
                "GPU score differs from CPU tolerance");
}

void test_ivf_roundtrip() {
    constexpr std::size_t n = 512;
    constexpr std::size_t nq = 8;
    constexpr std::size_t dim = 16;
    std::vector<float> database(n * dim);
    std::vector<float> queries(nq * dim);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t d = 0; d < dim; ++d)
            database[i * dim + d] = static_cast<float>(i % 16) * 4.0f +
                                    static_cast<float>((i + d) % 7) * 0.01f;
    for (std::size_t q = 0; q < nq; ++q)
        std::copy_n(database.data() + q * 16 * dim, dim, queries.data() + q * dim);
    gvs::IvfFlatIndex index(dim, 16, gvs::Metric::L2);
    index.train(database, n, 8, 7);
    index.add(database, n);
    index.save("gvs_test_index.bin");
    auto loaded = gvs::IvfFlatIndex::load("gvs_test_index.bin");
    const auto exact = gvs::cpu_exact_search(database, queries, n, nq, dim,
                                             gvs::Metric::L2, 10);
    const auto approximate = loaded.search(queries, nq, 4, gvs::SearchConfig{10, 4});
    require(gvs::recall_at_k(exact, approximate) >= 0.90,
            "IVF recall is unexpectedly low on clustered test data");
}

}  // namespace

int main() {
    try {
        test_metric(gvs::Metric::L2);
        test_metric(gvs::Metric::InnerProduct);
        test_ivf_roundtrip();
        std::cout << "all correctness tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
