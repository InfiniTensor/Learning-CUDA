#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "file_io.h"

namespace {

template <typename T>
void writeValue(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Operation>
void requireThrows(Operation operation, const std::string& message) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

void writeDatabase(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  const std::array<char, 8> magic = {'P', 'P', 'V', 'E', 'C', '0', '0', '1'};
  const std::uint64_t rows = 2;
  const std::uint32_t dim = 2;
  const std::uint32_t dtype = static_cast<std::uint32_t>(DataType::kFloat16);
  const std::uint32_t metric = static_cast<std::uint32_t>(Metric::kL2);
  const std::array<std::uint16_t, 4> values = {0x3c00, 0x0000, 0x4000, 0xbc00};
  output.write(magic.data(), magic.size());
  writeValue(output, rows);
  writeValue(output, dim);
  writeValue(output, dtype);
  writeValue(output, metric);
  output.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(values[0]));
}

void writeQueries(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  const std::array<char, 8> magic = {'P', 'P', 'Q', 'R', 'Y', '0', '0', '1'};
  const std::uint64_t rows = 1;
  const std::uint32_t dim = 2;
  const std::uint32_t dtype = static_cast<std::uint32_t>(DataType::kFloat16);
  const std::array<std::uint16_t, 2> values = {0x3c00, 0x0000};
  output.write(magic.data(), magic.size());
  writeValue(output, rows);
  writeValue(output, dim);
  writeValue(output, dtype);
  output.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(values[0]));
}

}  // namespace

int main() {
  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("purplepulse-file-io-test-" + std::to_string(unique));
  try {
    std::filesystem::create_directories(directory);
    const std::filesystem::path database_path = directory / "database.bin";
    const std::filesystem::path query_path = directory / "queries.bin";
    writeDatabase(database_path);
    writeQueries(query_path);

    const VectorDatabase database = readVectorDatabase(database_path.string());
    const QuerySet queries = readQuerySet(query_path.string());
    require(database.dtype == DataType::kFloat16, "database dtype 错误");
    require(queries.dtype == DataType::kFloat16, "query dtype 错误");
    require(database.values.empty() && queries.values.empty(),
            "FP16 文件不应展开为 FP32 存储");
    require(
        database.half_values.size() == 4 && database.half_values[2] == 0x4000,
        "database half 位模式错误");
    require(queries.half_values.size() == 2 && queries.half_values[0] == 0x3c00,
            "query half 位模式错误");
    SearchParams params;
    params.top_k = 1;
    validateInputs(database, queries, params);
    params.search_mode = "ivf_flat";
    bool rejected_unimplemented_ivf = false;
    try {
      validateInputs(database, queries, params);
    } catch (const std::runtime_error&) {
      rejected_unimplemented_ivf = true;
    }
    require(rejected_unimplemented_ivf,
            "未实现的 IVF-Flat 不应静默执行 exact search");

    const std::filesystem::path metadata_path = directory / "memory.bin";
    MemoryMetadata written_metadata;
    written_metadata.num_vectors = 3;
    written_metadata.timestamps = {100, 200, 300};
    written_metadata.importance = {0.1F, 0.5F, 1.0F};
    written_metadata.session_ids = {1, 1, 2};
    written_metadata.source_types = {0, 2, 1};
    writeMemoryMetadata(metadata_path.string(), written_metadata);
    const MemoryMetadata loaded_metadata =
        readMemoryMetadata(metadata_path.string());
    require(loaded_metadata.timestamps == written_metadata.timestamps &&
                loaded_metadata.importance == written_metadata.importance &&
                loaded_metadata.session_ids == written_metadata.session_ids &&
                loaded_metadata.source_types == written_metadata.source_types,
            "记忆元数据保存/加载后内容变化");

    const std::filesystem::path params_path = directory / "adaptive.conf";
    {
      std::ofstream output(params_path);
      output << "top_k = 10\n"
                "search_mode = ivf_flat\n"
                "exact_query_tile = 2\n"
                "exact_fused_layout = warp_query\n"
                "exact_fused_blocks = 96\n"
                "nprobe = 224\n"
                "nprobe_policy = score_mass\n"
                "adaptive_execution = grouped\n"
                "adaptive_nprobe_min = 64\n"
                "adaptive_nprobe_step = 16\n"
                "adaptive_target_mass = 0.85\n"
                "adaptive_temperature = 0.2\n"
                "memory_mode = fused\n"
                "memory_importance_weight = 0.15\n"
                "memory_recency_weight = 0.1\n"
                "memory_time_scale = 100\n"
                "memory_now = 300\n"
                "filter_session_id = any\n"
                "filter_source_type = 2\n";
    }
    const SearchParams adaptive = readSearchParams(params_path.string());
    require(adaptive.nprobe_policy == "score_mass" &&
                adaptive.exact_query_tile == 2 &&
                adaptive.exact_fused_layout == "warp_query" &&
                adaptive.exact_fused_blocks == 96 &&
                adaptive.adaptive_execution == "grouped" &&
                adaptive.adaptive_nprobe_min == 64 &&
                adaptive.adaptive_nprobe_step == 16 &&
                adaptive.adaptive_target_mass == 0.85F &&
                adaptive.adaptive_temperature == 0.2F &&
                adaptive.memory_mode == "fused" &&
                adaptive.memory_importance_weight == 0.15F &&
                adaptive.memory_now == 300 &&
                adaptive.filter_session_id == UINT32_MAX &&
                adaptive.filter_source_type == 2,
            "自适应 nprobe 或记忆参数解析错误");

    const std::filesystem::path bad_magic_path = directory / "bad-magic.bin";
    {
      std::ofstream output(bad_magic_path, std::ios::binary);
      output << "NOTAVECTORFILE";
    }
    requireThrows([&] { (void)readVectorDatabase(bad_magic_path.string()); },
                  "错误的向量库 magic/version 应被拒绝");

    const std::filesystem::path truncated_path = directory / "truncated.bin";
    writeQueries(truncated_path);
    std::filesystem::resize_file(
        truncated_path, std::filesystem::file_size(truncated_path) - 1);
    requireThrows([&] { (void)readQuerySet(truncated_path.string()); },
                  "截断的查询文件应被拒绝");

    const std::filesystem::path extra_bytes_path = directory / "extra.bin";
    writeDatabase(extra_bytes_path);
    {
      std::ofstream output(extra_bytes_path, std::ios::binary | std::ios::app);
      output.put('\0');
    }
    requireThrows([&] { (void)readVectorDatabase(extra_bytes_path.string()); },
                  "包含额外字节的向量库文件应被拒绝");

    std::filesystem::remove_all(directory);
    std::cout << "FP16 文件读取与损坏文件测试通过\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(directory);
    std::cerr << "测试失败: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
