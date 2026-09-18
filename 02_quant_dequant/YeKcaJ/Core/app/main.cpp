// C++ 程序入口：仅分发命令、设置打印精度和统一报告异常。
// GPU 算子在 kernels/，启动流程在 runtime/workspace.cu。
#include "app/commands.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace pipeline;
  try {
    std::cout << std::setprecision(10);
    const std::string command = argc > 1 ? argv[1] : "--help";
    if (command == "--help") {
      std::cout
          << "文件任务: pipeline input packed output block|tensor nearest|stochastic "
             "fp32|fp16|bf16 block_size seed verify|noverify\n"
          << "反量化:   pipeline --dequant-file packed output fp32|fp16|bf16\n"
          << "性能测试: pipeline --benchmark elements repeats [fixed_input.fp32]\n"
          << "固定输入: pipeline --benchmark-export elements output.fp32\n";
#ifdef LP_ENABLE_TESTS
      std::cout << "正确性:   pipeline --self-test [reference_directory]\n";
#endif
      return 0;
    }
    if ((argc == 4 || argc == 5) && command == "--benchmark") {
      benchmark(std::stoull(argv[2]), std::stoi(argv[3]), argc == 5 ? argv[4] : "");
      return 0;
    }
    if (argc == 4 && command == "--benchmark-export") {
      export_benchmark_input(std::stoull(argv[2]), argv[3]);
      return 0;
    }
    if ((argc == 2 || argc == 3) && command == "--self-test") {
#ifdef LP_ENABLE_TESTS
      self_test(argc == 3 ? argv[2] : LP_REFERENCE_DIR);
      return 0;
#else
      throw std::runtime_error("self-test requires a BUILD_TESTING=ON build");
#endif
    }
    if (argc == 5 && command == "--dequant-file") {
      restore_file(argv[2], argv[3], argv[4]);
      return 0;
    }
    run_pipeline(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "pipeline=FAIL " << e.what() << '\n';
    return 1;
  }
}
