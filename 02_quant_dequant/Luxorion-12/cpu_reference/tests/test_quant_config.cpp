#include "quant_config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
  namespace fs = std::filesystem;
  const fs::path path =
      fs::temp_directory_path() / "luxorion_quant_config_test.cfg";
  try {
    {
      std::ofstream out(path);
      out << "format = \"nvfp4\"\nblock_size = 16\nscale_mode = \"block\"\n"
             "output_type = \"fp16\"\nrounding = \"nearest\"\ntarget_gpu = "
             "\"T4\"\n";
    }
    const auto config = luxorion::read_quant_config(path.string());
    if (config.format != luxorion::QuantFormat::Nvfp4 ||
        config.block_size != 16 ||
        config.output_type != luxorion::OutputType::FP16 ||
        config.target_gpu != "T4")
      return 1;
    {
      std::ofstream out(path);
      out << "format = \"mxfp8\"\nblock_size = 16\nscale_mode = \"block\"\n"
             "output_type = \"fp32\"\nrounding = \"nearest\"\ntarget_gpu = "
             "\"T4\"\n";
    }
    bool rejected = false;
    try {
      (void)luxorion::read_quant_config(path.string());
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    fs::remove(path);
    if (!rejected)
      return 1;
    std::cout
        << "Quant config: valid file parsed and invalid block size rejected\n";
    return 0;
  } catch (...) {
    fs::remove(path);
    throw;
  }
}
