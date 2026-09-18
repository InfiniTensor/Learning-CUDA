#pragma once

#include <cstddef>
#include <string>

namespace pipeline {

// 普通文件任务；参数协议由 Python 配置入口转换，不在这里暴露 GPU 实现。
void run_pipeline(int argc, char** argv);
void restore_file(const std::string& input, const std::string& output, const std::string& dtype);

// 独立的基准与自测入口，不参与正常文件任务。
void benchmark(std::size_t count, int repeats, const std::string& input_file = "");
void export_benchmark_input(std::size_t count, const std::string& output_file);
#ifdef LP_ENABLE_TESTS
void self_test(const std::string& reference_dir);
#endif

}  // namespace pipeline
