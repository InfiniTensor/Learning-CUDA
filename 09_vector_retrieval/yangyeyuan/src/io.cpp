#include "vector_search.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace vs {
namespace {

// 四种文件格式各自的 8 字节 magic,用于识别文件类型。
constexpr std::array<char, 8> kVectorMagic{'V', 'E', 'C', 'D', 'B', '0', '0', '1'};  // 向量库
constexpr std::array<char, 8> kQueryMagic{'V', 'E', 'C', 'Q', 'R', '0', '0', '1'};   // 查询集
constexpr std::array<char, 8> kResultMagic{'V', 'E', 'C', 'R', 'E', 'S', '0', '1'};  // 检索结果
constexpr std::array<char, 8> kIndexMagic{'I', 'V', 'F', 'F', 'L', 'T', '0', '1'};   // IVF 索引
constexpr std::uint32_t kVersion = 1;  // 格式版本,防止跨版本误读

// 写一个平凡可拷贝类型(固定字节序为 little-endian)。
template <typename T>
void write_scalar(std::ostream& out, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) throw std::runtime_error("failed to write binary file");
}

// 读一个标量;文件被截断时抛错。
template <typename T>
T read_scalar(std::istream& in) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("truncated binary file");
  return value;
}

// 连续写一个 vector 的原始字节。
template <typename T>
void write_array(std::ostream& out, const std::vector<T>& values) {
  if (values.empty()) return;
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) throw std::runtime_error("failed to write binary payload");
}

// 连续读一个 vector 的原始字节;不足即截断。
template <typename T>
void read_array(std::istream& in, std::vector<T>* values) {
  if (values->empty()) return;
  in.read(reinterpret_cast<char*>(values->data()),
          static_cast<std::streamsize>(values->size() * sizeof(T)));
  if (!in) throw std::runtime_error("truncated binary payload");
}

// 校验 count*dim 不溢出 host 地址空间,并返回元素总数。
std::size_t checked_elements(std::uint64_t count, std::uint32_t dim) {
  if (dim == 0 || count == 0) throw std::runtime_error("count and dim must be positive");
  if (count > std::numeric_limits<std::size_t>::max() / dim) {
    throw std::runtime_error("vector shape overflows host address space");
  }
  return static_cast<std::size_t>(count) * dim;
}

// 把 float 的位模式当成 uint32(用于比较/打包)。
std::uint32_t float_bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// 把 uint32 位模式还原成 float。
float bits_float(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// FP16 → FP32 转换,处理 subnormal、Inf/NaN。
// FP16 布局: sign(1) | exponent(5) | mantissa(10)。
float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exponent = (h >> 10) & 0x1fu;
  std::uint32_t mantissa = h & 0x03ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // ±0
    } else {
      // subnormal: 逐位左移找到最高位,再映射到 FP32 的 exponent。
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      bits = sign | static_cast<std::uint32_t>(127 - 14 - shift) << 23 |
             mantissa << 13;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | mantissa << 13;  // Inf/NaN
  } else {
    bits = sign | (exponent + 112u) << 23 | mantissa << 13;  // 常规值: 指数偏移量 -15 → -127
  }
  return bits_float(bits);
}

// FP32 → FP16 转换,采用 round-to-nearest-even 舍入。
std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = float_bits(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t abs = bits & 0x7fffffffu;
  if (abs >= 0x7f800000u) {
    // Inf/NaN:NaN 的 mantissa 非 0,保留部分 bit。
    return static_cast<std::uint16_t>(sign | (abs > 0x7f800000u ? 0x7e00u : 0x7c00u));
  }
  // 重算指数:FP32 偏移 127 → FP16 偏移 15。
  const int exponent = static_cast<int>((abs >> 23) & 0xffu) - 127 + 15;
  const std::uint32_t mantissa = abs & 0x7fffffu;
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);  // 溢出 → Inf
  if (exponent <= 0) {
    // 下溢:进入 subnormal 或 0 区间。
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t hidden = mantissa | 0x800000u;
    const int shift = 14 - exponent;
    std::uint32_t rounded = hidden >> shift;
    const std::uint32_t remainder = hidden & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    // 银行家舍入:恰好在中间时往偶数方向走。
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  // 常规值:进位逻辑 + 低 13 位作舍入判断。
  std::uint32_t rounded = mantissa + 0x0fffu + ((mantissa >> 13) & 1u);
  std::uint32_t half_exp = static_cast<std::uint32_t>(exponent);
  if (rounded & 0x800000u) {
    // mantissa 进位溢出了尾数位,指数 +1。
    rounded = 0;
    if (++half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  return static_cast<std::uint16_t>(sign | half_exp << 10 | rounded >> 13);
}

// 打开输出文件:自动创建父目录,truncate 已存在的文件。
std::ofstream open_output(const std::string& path) {
  const std::filesystem::path file(path);
  if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open output file: " + path);
  return out;
}

// 打开输入文件。
std::ifstream open_input(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open input file: " + path);
  return in;
}

// 校验 8 字节 magic 是否匹配;不匹配说明文件类型或版本不对。
void expect_magic(std::istream& in, const std::array<char, 8>& expected) {
  std::array<char, 8> actual{};
  in.read(actual.data(), actual.size());
  if (!in || actual != expected) throw std::runtime_error("invalid binary magic");
}

// 去除首尾空白。
std::string trim(std::string value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

// 解析非负整数,拒绝空串/负号/多余字符。
std::uint64_t parse_unsigned(const std::string& key, const std::string& value) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error("invalid unsigned integer for " + key);
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size()) throw std::runtime_error("invalid integer for " + key);
  return parsed;
}

}  // namespace

// 读取向量库(is_query=false)或查询集(is_query=true)文件。
// FP16 文件在读取时统一转成 FP32 存进 VectorDataset.values。
VectorDataset load_vectors(const std::string& path, bool is_query) {
  auto in = open_input(path);
  expect_magic(in, is_query ? kQueryMagic : kVectorMagic);
  const auto version = read_scalar<std::uint32_t>(in);
  const auto dtype_raw = read_scalar<std::uint32_t>(in);
  const auto metric_raw = read_scalar<std::uint32_t>(in);
  const auto dim = read_scalar<std::uint32_t>(in);
  const auto count = read_scalar<std::uint64_t>(in);
  const auto payload_bytes = read_scalar<std::uint64_t>(in);
  if (version != kVersion) throw std::runtime_error("unsupported vector file version");
  if (dtype_raw != static_cast<std::uint32_t>(DType::kFp32) &&
      dtype_raw != static_cast<std::uint32_t>(DType::kFp16)) {
    throw std::runtime_error("unsupported vector dtype");
  }
  // 查询集不校验 metric(其 metric 字段写 0,由索引继承)。
  if (!is_query && metric_raw != static_cast<std::uint32_t>(Metric::kL2)) {
    throw std::runtime_error("only l2 metric is supported");
  }
  const auto dtype = static_cast<DType>(dtype_raw);
  const std::size_t elements = checked_elements(count, dim);
  const std::uint64_t expected_bytes = elements * (dtype == DType::kFp32 ? 4ull : 2ull);
  if (payload_bytes != expected_bytes) throw std::runtime_error("vector payload size mismatch");

  VectorDataset result;
  result.count = count;
  result.dim = dim;
  result.source_dtype = dtype;
  result.metric = Metric::kL2;
  result.values.resize(elements);
  if (dtype == DType::kFp32) {
    read_array(in, &result.values);
  } else {
    // FP16:先读原始 half,再逐元素转 FP32。
    std::vector<std::uint16_t> halves(elements);
    read_array(in, &halves);
    std::transform(halves.begin(), halves.end(), result.values.begin(), half_to_float);
  }
  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing bytes in vector file");
  }
  return result;
}

// 写入向量库(is_query=false)或查询集(is_query=true)文件。
// dtype 决定写 FP32 还是 FP16;写 FP16 时做一次有损转换。
void save_vectors(const std::string& path, const VectorDataset& data,
                  bool is_query, DType dtype) {
  const std::size_t elements = checked_elements(data.count, data.dim);
  if (data.values.size() != elements) throw std::runtime_error("invalid vector payload shape");
  auto out = open_output(path);
  const auto& magic = is_query ? kQueryMagic : kVectorMagic;
  out.write(magic.data(), magic.size());
  write_scalar(out, kVersion);
  write_scalar(out, static_cast<std::uint32_t>(dtype));
  write_scalar(out, is_query ? 0u : static_cast<std::uint32_t>(data.metric));
  write_scalar(out, data.dim);
  write_scalar(out, data.count);
  write_scalar(out, static_cast<std::uint64_t>(elements) *
                        (dtype == DType::kFp32 ? 4ull : 2ull));
  if (dtype == DType::kFp32) {
    write_array(out, data.values);
  } else {
    std::vector<std::uint16_t> halves(elements);
    std::transform(data.values.begin(), data.values.end(), halves.begin(), float_to_half);
    write_array(out, halves);
  }
}

// 解析极简的 key=value 配置文件(支持 # 注释和双引号字符串)。
SearchConfig load_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  SearchConfig config;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    const auto comment = line.find('#');  // 去掉注释
    if (comment != std::string::npos) line.erase(comment);
    line = trim(line);
    if (line.empty()) continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error("config line missing '=' at line " + std::to_string(line_number));
    }
    const std::string key = trim(line.substr(0, equal));
    std::string value = trim(line.substr(equal + 1));
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
      value = value.substr(1, value.size() - 2);  // 去掉引号
    }
    const auto number = [&] { return parse_unsigned(key, value); };
    const auto number32 = [&] {
      const auto parsed = number();
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("integer out of uint32 range for " + key);
      }
      return static_cast<std::uint32_t>(parsed);
    };
    if (key == "top_k") config.top_k = number32();
    else if (key == "search_mode") config.search_mode = value;
    else if (key == "batch_size") config.batch_size = number32();
    else if (key == "nlist") config.nlist = number32();
    else if (key == "nprobe") config.nprobe = number32();
    else if (key == "kmeans_iters") config.kmeans_iters = number32();
    else if (key == "train_size") config.train_size = number32();
    else if (key == "seed") config.seed = number();
    else throw std::runtime_error("unknown config key: " + key);
  }
  if (config.top_k == 0 || config.batch_size == 0 || config.nlist == 0 ||
      config.nprobe == 0 || config.kmeans_iters == 0 || config.train_size == 0) {
    throw std::runtime_error("all numeric config values must be positive");
  }
  if (config.search_mode != "exact" && config.search_mode != "ivf_flat") {
    throw std::runtime_error("search_mode must be exact or ivf_flat");
  }
  return config;
}

// 写出检索结果文件:头部元信息 + ids + scores。
void save_results(const std::string& path, const SearchResult& result) {
  const std::size_t elements = checked_elements(result.num_queries, result.top_k);
  if (result.ids.size() != elements || result.scores.size() != elements) {
    throw std::runtime_error("invalid result shape");
  }
  auto out = open_output(path);
  out.write(kResultMagic.data(), kResultMagic.size());
  write_scalar(out, kVersion);
  write_scalar(out, static_cast<std::uint32_t>(result.metric));
  write_scalar(out, result.top_k);
  write_scalar(out, 0u);  // reserved
  write_scalar(out, result.num_queries);
  write_array(out, result.ids);
  write_array(out, result.scores);
}

// 读取检索结果文件。
SearchResult load_results(const std::string& path) {
  auto in = open_input(path);
  expect_magic(in, kResultMagic);
  if (read_scalar<std::uint32_t>(in) != kVersion) throw std::runtime_error("unsupported result version");
  SearchResult result;
  result.metric = static_cast<Metric>(read_scalar<std::uint32_t>(in));
  result.top_k = read_scalar<std::uint32_t>(in);
  (void)read_scalar<std::uint32_t>(in);  // reserved
  result.num_queries = read_scalar<std::uint64_t>(in);
  const auto elements = checked_elements(result.num_queries, result.top_k);
  result.ids.resize(elements);
  result.scores.resize(elements);
  read_array(in, &result.ids);
  read_array(in, &result.scores);
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing result bytes");
  return result;
}

// 写出 IVF 索引:质心、桶偏移、原始 ID、桶内重排的维度主序向量。
// 保存前校验索引形状(offsets 从 0 开始、到 count 结束、单调递增)。
void save_index(const std::string& path, const IvfIndexData& index) {
  const auto vector_elements = checked_elements(index.count, index.dim);
  if (index.nlist == 0 || index.centroids.size() != static_cast<std::size_t>(index.nlist) * index.dim ||
      index.offsets.size() != static_cast<std::size_t>(index.nlist) + 1 ||
      index.original_ids.size() != index.count || index.vectors_soa.size() != vector_elements ||
      index.offsets.front() != 0 || index.offsets.back() != index.count ||
      !std::is_sorted(index.offsets.begin(), index.offsets.end())) {
    throw std::runtime_error("invalid IVF index shape");
  }
  auto out = open_output(path);
  out.write(kIndexMagic.data(), kIndexMagic.size());
  write_scalar(out, kVersion);
  write_scalar(out, static_cast<std::uint32_t>(DType::kFp32));
  write_scalar(out, static_cast<std::uint32_t>(index.metric));
  write_scalar(out, index.dim);
  write_scalar(out, index.nlist);
  write_scalar(out, 0u);  // reserved
  write_scalar(out, index.count);
  // payload 总字节数,加载时用它做一致性校验。
  const std::uint64_t payload = index.centroids.size() * sizeof(float) +
                                index.offsets.size() * sizeof(std::uint64_t) +
                                index.original_ids.size() * sizeof(std::uint32_t) +
                                index.vectors_soa.size() * sizeof(float);
  write_scalar(out, payload);
  write_array(out, index.centroids);
  write_array(out, index.offsets);
  write_array(out, index.original_ids);
  write_array(out, index.vectors_soa);
}

// 读取 IVF 索引,做与保存时对称的完整性校验。
IvfIndexData load_index(const std::string& path) {
  auto in = open_input(path);
  expect_magic(in, kIndexMagic);
  if (read_scalar<std::uint32_t>(in) != kVersion) throw std::runtime_error("unsupported index version");
  if (read_scalar<std::uint32_t>(in) != static_cast<std::uint32_t>(DType::kFp32)) {
    throw std::runtime_error("unsupported index storage dtype");
  }
  IvfIndexData index;
  index.metric = static_cast<Metric>(read_scalar<std::uint32_t>(in));
  index.dim = read_scalar<std::uint32_t>(in);
  index.nlist = read_scalar<std::uint32_t>(in);
  (void)read_scalar<std::uint32_t>(in);  // reserved
  index.count = read_scalar<std::uint64_t>(in);
  const auto payload = read_scalar<std::uint64_t>(in);
  if (index.metric != Metric::kL2 || index.nlist == 0 || index.nlist > index.count) {
    throw std::runtime_error("invalid IVF index metadata");
  }
  const auto vector_elements = checked_elements(index.count, index.dim);
  const std::uint64_t expected = static_cast<std::uint64_t>(index.nlist) * index.dim * sizeof(float) +
                                 (static_cast<std::uint64_t>(index.nlist) + 1) * sizeof(std::uint64_t) +
                                 index.count * sizeof(std::uint32_t) +
                                 static_cast<std::uint64_t>(vector_elements) * sizeof(float);
  if (payload != expected) throw std::runtime_error("IVF index payload size mismatch");
  index.centroids.resize(static_cast<std::size_t>(index.nlist) * index.dim);
  index.offsets.resize(static_cast<std::size_t>(index.nlist) + 1);
  index.original_ids.resize(index.count);
  index.vectors_soa.resize(vector_elements);
  read_array(in, &index.centroids);
  read_array(in, &index.offsets);
  read_array(in, &index.original_ids);
  read_array(in, &index.vectors_soa);
  if (index.offsets.front() != 0 || index.offsets.back() != index.count ||
      !std::is_sorted(index.offsets.begin(), index.offsets.end())) {
    throw std::runtime_error("corrupt IVF offsets");
  }
  for (const auto id : index.original_ids) {
    if (id >= index.count) throw std::runtime_error("corrupt IVF original id");
  }
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing index bytes");
  return index;
}

// 写出性能日志(手拼 JSON,无第三方库依赖)。
void save_performance_log(const std::string& path, const SearchConfig& config,
                          const SearchMetrics& metrics) {
  auto out = open_output(path);
  out << std::fixed << std::setprecision(6)
      << "{\n  \"search_mode\": \"" << config.search_mode << "\",\n"
      << "  \"top_k\": " << config.top_k << ",\n"
      << "  \"batch_size\": " << config.batch_size << ",\n"
      << "  \"nlist\": " << config.nlist << ",\n"
      << "  \"nprobe\": " << config.nprobe << ",\n"
      << "  \"build_ms\": " << metrics.build_ms << ",\n"
      << "  \"total_ms\": " << metrics.total_ms << ",\n"
      << "  \"gpu_ms\": " << metrics.gpu_ms << ",\n"
      << "  \"qps\": " << metrics.qps << ",\n"
      << "  \"p50_batch_ms\": " << metrics.p50_ms << ",\n"
      << "  \"p99_batch_ms\": " << metrics.p99_ms << ",\n"
      << "  \"cpu_baseline_ms\": " << metrics.cpu_baseline_ms << ",\n"
      << "  \"speedup_vs_cpu\": " << metrics.speedup_vs_cpu << ",\n"
      << "  \"peak_device_bytes\": " << metrics.peak_device_bytes << "\n}\n";
}

// 写出质量日志(recall 等)。
void save_quality_log(const std::string& path, const SearchConfig& config,
                      const QualityMetrics& metrics) {
  auto out = open_output(path);
  out << std::fixed << std::setprecision(8)
      << "{\n  \"search_mode\": \"" << config.search_mode << "\",\n"
      << "  \"top_k\": " << config.top_k << ",\n"
      << "  \"batch_size\": " << config.batch_size << ",\n"
      << "  \"nprobe\": " << config.nprobe << ",\n"
      << "  \"recall_at_k\": " << metrics.recall_at_k << ",\n"
      << "  \"mean_distance_error\": " << metrics.mean_distance_error << "\n}\n";
}

}  // namespace vs
