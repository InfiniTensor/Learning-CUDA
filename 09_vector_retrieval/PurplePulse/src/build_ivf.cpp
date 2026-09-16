#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include "file_io.h"
#include "ivf_flat.h"

namespace {

void printUsage(const char* program) {
  std::cerr << "用法:\n  " << program
            << " --database <文件> --output <索引文件> --nlist <数量>"
               " [--iterations 15] [--training-samples 100000]"
               " [--threads N]\n";
}

std::unordered_map<std::string, std::string> parseArguments(int argc,
                                                            char** argv) {
  std::unordered_map<std::string, std::string> arguments;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::runtime_error("命令行参数必须是 --名称 值 的形式");
    }
    arguments[argv[index]] = argv[index + 1];
  }
  for (const char* required : {"--database", "--output", "--nlist"}) {
    if (arguments.count(required) == 0) {
      throw std::runtime_error(std::string("缺少参数: ") + required);
    }
  }
  return arguments;
}

std::uint64_t readPositive(
    const std::unordered_map<std::string, std::string>& arguments,
    const std::string& name, std::uint64_t default_value) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  std::size_t parsed_chars = 0;
  const unsigned long long parsed = std::stoull(found->second, &parsed_chars);
  if (parsed_chars != found->second.size() || parsed == 0) {
    throw std::runtime_error(name + " 必须是正整数");
  }
  return parsed;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    }
    if (argc == 1) {
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
    const auto arguments = parseArguments(argc, argv);
    const std::uint64_t parsed_nlist = readPositive(arguments, "--nlist", 0);
    const std::uint64_t parsed_iterations =
        readPositive(arguments, "--iterations", 15);
    if (parsed_nlist > UINT32_MAX || parsed_iterations > UINT32_MAX) {
      throw std::runtime_error("nlist 或 iterations 超过 uint32 范围");
    }
    const std::uint64_t training_samples =
        readPositive(arguments, "--training-samples", 100000);
    const std::uint32_t default_threads = std::max(
        1U, std::min(16U, std::thread::hardware_concurrency()));
    const std::uint64_t parsed_threads =
        readPositive(arguments, "--threads", default_threads);
    if (parsed_threads > 1024) {
      throw std::runtime_error("threads 必须位于 [1, 1024]");
    }
    const VectorDatabase database =
        readVectorDatabase(arguments.at("--database"));

    const auto build_start = std::chrono::steady_clock::now();
    const IvfFlatIndex index = buildIvfFlatIndex(
        database, static_cast<std::uint32_t>(parsed_nlist),
        static_cast<std::uint32_t>(parsed_iterations), training_samples,
        static_cast<std::uint32_t>(parsed_threads));
    const auto build_end = std::chrono::steady_clock::now();

    const std::filesystem::path output_path(arguments.at("--output"));
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    const auto save_start = std::chrono::steady_clock::now();
    writeIvfFlatIndex(output_path.string(), index);
    const auto save_end = std::chrono::steady_clock::now();

    std::uint64_t minimum_bucket = index.num_vectors;
    std::uint64_t maximum_bucket = 0;
    std::uint32_t empty_buckets = 0;
    for (std::uint32_t center_id = 0; center_id < index.nlist; ++center_id) {
      const std::uint64_t size =
          index.offsets[center_id + 1] - index.offsets[center_id];
      minimum_bucket = std::min(minimum_bucket, size);
      maximum_bucket = std::max(maximum_bucket, size);
      empty_buckets += size == 0;
    }
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start)
            .count();
    const double save_ms =
        std::chrono::duration<double, std::milli>(save_end - save_start)
            .count();
    std::cout << "IVF-Flat 索引构建完成" << "\n向量库: " << index.num_vectors
              << " x " << index.dim
              << "\ndtype/metric: " << dataTypeName(index.dtype) << '/'
              << metricName(index.metric) << "\nnlist: " << index.nlist
              << "\n迭代/训练样本: " << parsed_iterations << '/'
              << std::min<std::uint64_t>(training_samples, index.num_vectors)
              << "\n建库线程: " << parsed_threads
              << "\n构建时间: " << build_ms << " ms"
              << "\n保存时间: " << save_ms << " ms"
              << "\n桶大小 min/avg/max: " << minimum_bucket << '/'
              << static_cast<double>(index.num_vectors) / index.nlist << '/'
              << maximum_bucket << "\n空桶: " << empty_buckets
              << "\n索引大小: " << std::filesystem::file_size(output_path)
              << " bytes\n输出: " << output_path << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << '\n';
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
}
