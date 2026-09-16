#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vector_types.h"

IvfFlatIndex buildIvfFlatIndex(const VectorDatabase& database,
                               std::uint32_t nlist,
                               std::uint32_t iterations = 15,
                               std::uint64_t max_training_vectors = 100000,
                               std::uint32_t num_threads = 1);
void writeIvfFlatIndex(const std::string& path, const IvfFlatIndex& index);
IvfFlatIndex readIvfFlatIndex(const std::string& path);
void validateIvfFlatIndex(const IvfFlatIndex& index);

std::vector<std::vector<std::uint64_t>> selectIvfCandidatePositions(
    const IvfFlatIndex& index, const QuerySet& queries, std::uint32_t nprobe);

SearchResults cpuIvfFlatSearch(const IvfFlatIndex& index,
                               const QuerySet& queries,
                               const SearchParams& params,
                               const MemoryMetadata* metadata = nullptr);

// GPU IVF 执行器：中心、桶 offsets、索引向量和 ID 常驻显存；GPU 完成
// 中心选择、各桶候选扫描、局部 Top-K 和最终归并。
class GpuIvfFlatSearchEngine {
 public:
  GpuIvfFlatSearchEngine(const IvfFlatIndex& index,
                         const QuerySet& initial_queries,
                         const SearchParams& params,
                         const MemoryMetadata* metadata = nullptr);
  ~GpuIvfFlatSearchEngine();

  GpuIvfFlatSearchEngine(const GpuIvfFlatSearchEngine&) = delete;
  GpuIvfFlatSearchEngine& operator=(const GpuIvfFlatSearchEngine&) = delete;
  GpuIvfFlatSearchEngine(GpuIvfFlatSearchEngine&&) noexcept;
  GpuIvfFlatSearchEngine& operator=(GpuIvfFlatSearchEngine&&) noexcept;

  SearchResults search(const QuerySet& queries, SearchStats* stats = nullptr);
  double indexH2DMilliseconds() const;
  std::size_t deviceBytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

SearchResults gpuIvfFlatSearch(const IvfFlatIndex& index,
                               const QuerySet& queries,
                               const SearchParams& params,
                               SearchStats* stats = nullptr,
                               const MemoryMetadata* metadata = nullptr);

double recallAtK(const SearchResults& exact, const SearchResults& approximate);
