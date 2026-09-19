#include "vector_search/search.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace vector_search {
namespace {

constexpr std::array<char, 8> kDatabaseMagic = {'V', 'E', 'C', 'D', 'B', '0',
                                                 '1', '\0'};
constexpr std::array<char, 8> kQueryMagic = {'V', 'E', 'C', 'Q', 'R', 'Y', '1',
                                              '\0'};
constexpr std::array<char, 8> kIndexMagic = {'I', 'V', 'F', 'F', 'L', 'T', '1',
                                              '\0'};
constexpr std::uint32_t kVersion = 1;

void require_little_endian_host() {
  const std::uint16_t marker = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&marker) != 1) {
    throw std::runtime_error("binary format currently requires a little-endian host");
  }
}

template <typename T>
T read_pod(std::istream& input, const char* name) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) throw std::runtime_error(std::string("truncated field: ") + name);
  return value;
}

template <typename T>
void write_pod(std::ostream& output, T value, const char* name) {
  static_assert(std::is_trivially_copyable_v<T>);
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!output) throw std::runtime_error(std::string("failed to write: ") + name);
}

std::array<char, 8> read_magic(std::istream& input) {
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (!input) throw std::runtime_error("truncated binary magic");
  return magic;
}

void write_magic(std::ostream& output, const std::array<char, 8>& magic) {
  output.write(magic.data(), magic.size());
  if (!output) throw std::runtime_error("failed to write binary magic");
}

std::size_t checked_elements(std::int64_t count, std::int32_t dim) {
  if (count <= 0 || dim <= 0) {
    throw std::runtime_error("count and dim must be positive");
  }
  const auto unsigned_count = static_cast<std::uint64_t>(count);
  const auto unsigned_dim = static_cast<std::uint64_t>(dim);
  if (unsigned_count > std::numeric_limits<std::size_t>::max() / unsigned_dim) {
    throw std::runtime_error("tensor element count overflows size_t");
  }
  return static_cast<std::size_t>(unsigned_count * unsigned_dim);
}

float half_to_float(std::uint16_t value) {
  // IEEE-754 binary16 转换，不依赖 CUDA。
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      const std::uint32_t float_exponent =
          static_cast<std::uint32_t>(127 - 15 + 1 - shift);
      bits = sign | (float_exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    exponent += static_cast<std::uint32_t>(127 - 15);
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::vector<float> read_values(std::istream& input, std::size_t elements,
                               DType dtype) {
  std::vector<float> values(elements);
  if (dtype == DType::kFloat32) {
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(elements * sizeof(float)));
  } else {
    std::vector<std::uint16_t> half(elements);
    input.read(reinterpret_cast<char*>(half.data()),
               static_cast<std::streamsize>(elements * sizeof(std::uint16_t)));
    if (input) {
      for (std::size_t index = 0; index < elements; ++index) {
        values[index] = half_to_float(half[index]);
      }
    }
  }
  if (!input) throw std::runtime_error("truncated vector payload");
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing bytes in vector file");
  }
  return values;
}

DType checked_dtype(std::uint8_t value) {
  if (value == static_cast<std::uint8_t>(DType::kFloat32)) {
    return DType::kFloat32;
  }
  if (value == static_cast<std::uint8_t>(DType::kFloat16)) {
    return DType::kFloat16;
  }
  throw std::runtime_error("unsupported dtype enum in binary header");
}

Metric checked_metric(std::uint8_t value) {
  if (value == static_cast<std::uint8_t>(Metric::kL2)) return Metric::kL2;
  if (value == static_cast<std::uint8_t>(Metric::kInnerProduct)) {
    return Metric::kInnerProduct;
  }
  if (value == static_cast<std::uint8_t>(Metric::kCosine)) {
    return Metric::kCosine;
  }
  throw std::runtime_error("unsupported metric enum in binary header");
}

void ensure_parent(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
}

}  // namespace

VectorDatabase load_vector_database(const std::string& path) {
  // 校验文件头后解码为行主序 FP32 数组。
  require_little_endian_host();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open vector database: " + path);
  if (read_magic(input) != kDatabaseMagic) {
    throw std::runtime_error("invalid vector database magic");
  }
  if (read_pod<std::uint32_t>(input, "version") != kVersion) {
    throw std::runtime_error("unsupported vector database version");
  }
  VectorDatabase database;
  database.count = read_pod<std::int64_t>(input, "num_vectors");
  database.dim = read_pod<std::int32_t>(input, "dim");
  database.source_dtype =
      checked_dtype(read_pod<std::uint8_t>(input, "dtype"));
  database.metric = checked_metric(read_pod<std::uint8_t>(input, "metric"));
  (void)read_pod<std::uint16_t>(input, "reserved");
  database.values =
      read_values(input, checked_elements(database.count, database.dim),
                  database.source_dtype);
  return database;
}

QuerySet load_queries(const std::string& path) {
  // 查询文件不存储度量类型，搜索时使用数据库的度量。
  require_little_endian_host();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open query file: " + path);
  if (read_magic(input) != kQueryMagic) {
    throw std::runtime_error("invalid query-file magic");
  }
  if (read_pod<std::uint32_t>(input, "version") != kVersion) {
    throw std::runtime_error("unsupported query-file version");
  }
  QuerySet queries;
  queries.count = read_pod<std::int64_t>(input, "num_queries");
  queries.dim = read_pod<std::int32_t>(input, "dim");
  queries.source_dtype = checked_dtype(read_pod<std::uint8_t>(input, "dtype"));
  std::array<char, 3> reserved{};
  input.read(reserved.data(), reserved.size());
  if (!input) throw std::runtime_error("truncated query reserved bytes");
  queries.values = read_values(input, checked_elements(queries.count, queries.dim),
                               queries.source_dtype);
  return queries;
}

void save_ivf_index(const IvfIndex& index, const std::string& path) {
  // 索引保存中心、CSR 偏移和向量 ID；向量数据另存于数据库文件。
  require_little_endian_host();
  if (index.num_vectors <= 0 || index.dim <= 0 || index.nlist <= 0 ||
      index.centers.size() !=
          static_cast<std::size_t>(index.nlist) * index.dim ||
      index.offsets.size() != static_cast<std::size_t>(index.nlist + 1) ||
      index.ids.size() != static_cast<std::size_t>(index.num_vectors) ||
      index.offsets.back() != static_cast<std::uint64_t>(index.num_vectors)) {
    throw std::runtime_error("cannot save malformed IVF index");
  }
  ensure_parent(path);
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot write IVF index: " + path);
  write_magic(output, kIndexMagic);
  write_pod(output, kVersion, "version");
  write_pod(output, index.num_vectors, "num_vectors");
  write_pod(output, index.dim, "dim");
  write_pod(output, static_cast<std::int32_t>(index.nlist), "nlist");
  write_pod(output, static_cast<std::uint8_t>(index.metric), "metric");
  const std::array<char, 3> reserved{};
  output.write(reserved.data(), reserved.size());
  output.write(reinterpret_cast<const char*>(index.centers.data()),
               static_cast<std::streamsize>(index.centers.size() * sizeof(float)));
  output.write(reinterpret_cast<const char*>(index.offsets.data()),
               static_cast<std::streamsize>(index.offsets.size() *
                                            sizeof(std::uint64_t)));
  output.write(reinterpret_cast<const char*>(index.ids.data()),
               static_cast<std::streamsize>(index.ids.size() *
                                            sizeof(std::int64_t)));
  if (!output) throw std::runtime_error("failed while writing IVF payload");
}

IvfIndex load_ivf_index(const std::string& path) {
  // 校验偏移单调性、末端大小和 ID 范围。
  require_little_endian_host();
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open IVF index: " + path);
  if (read_magic(input) != kIndexMagic) {
    throw std::runtime_error("invalid IVF index magic");
  }
  if (read_pod<std::uint32_t>(input, "version") != kVersion) {
    throw std::runtime_error("unsupported IVF index version");
  }
  IvfIndex index;
  index.num_vectors = read_pod<std::int64_t>(input, "num_vectors");
  index.dim = read_pod<std::int32_t>(input, "dim");
  index.nlist = read_pod<std::int32_t>(input, "nlist");
  index.metric = checked_metric(read_pod<std::uint8_t>(input, "metric"));
  std::array<char, 3> reserved{};
  input.read(reserved.data(), reserved.size());
  if (!input) throw std::runtime_error("truncated IVF reserved bytes");
  checked_elements(index.num_vectors, index.dim);
  if (index.nlist <= 0 || index.nlist > index.num_vectors) {
    throw std::runtime_error("invalid nlist in IVF index");
  }
  index.centers.resize(static_cast<std::size_t>(index.nlist) * index.dim);
  index.offsets.resize(static_cast<std::size_t>(index.nlist + 1));
  index.ids.resize(static_cast<std::size_t>(index.num_vectors));
  input.read(reinterpret_cast<char*>(index.centers.data()),
             static_cast<std::streamsize>(index.centers.size() * sizeof(float)));
  input.read(reinterpret_cast<char*>(index.offsets.data()),
             static_cast<std::streamsize>(index.offsets.size() *
                                          sizeof(std::uint64_t)));
  input.read(reinterpret_cast<char*>(index.ids.data()),
             static_cast<std::streamsize>(index.ids.size() *
                                          sizeof(std::int64_t)));
  if (!input || input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("truncated or oversized IVF index payload");
  }
  if (index.offsets.front() != 0 ||
      index.offsets.back() != static_cast<std::uint64_t>(index.num_vectors)) {
    throw std::runtime_error("invalid IVF offsets");
  }
  for (int list = 0; list < index.nlist; ++list) {
    if (index.offsets[list] > index.offsets[list + 1]) {
      throw std::runtime_error("IVF offsets are not monotonic");
    }
  }
  for (const auto id : index.ids) {
    if (id < 0 || id >= index.num_vectors) {
      throw std::runtime_error("IVF index contains out-of-range vector id");
    }
  }
  return index;
}

void write_search_results(const SearchResults& results,
                          const std::string& path) {
  if (results.num_queries < 0 || results.top_k <= 0 ||
      results.ids.size() !=
          static_cast<std::size_t>(results.num_queries) * results.top_k ||
      results.scores.size() != results.ids.size()) {
    throw std::runtime_error("cannot write malformed search results");
  }
  ensure_parent(path);
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write results: " + path);
  output << std::setprecision(9);
  for (std::int64_t query = 0; query < results.num_queries; ++query) {
    output << query;
    for (int rank = 0; rank < results.top_k; ++rank) {
      const auto offset = static_cast<std::size_t>(query) * results.top_k + rank;
      output << ' ' << results.ids[offset] << ':' << results.scores[offset];
    }
    output << '\n';
  }
}

}  // namespace vector_search
