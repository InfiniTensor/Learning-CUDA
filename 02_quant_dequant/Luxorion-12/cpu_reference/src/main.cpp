/*
 * 模块用途：作为CPU教学演示入口，将独立模块连接成可以直接运行、观察的流程。
 * 输入：程序内置的两份固定FP32样例；命令行可指定CSV输出目录。
 * 输出：中文终端日志，以及输入、编码、scale、还原值、指标等演示CSV。
 * 主流程：创建样例 -> CPU量化 -> CPU反量化 -> 指标计算 -> 展示/导出。
 * 职责边界：负责组织调用与解释结果；低精度计算由formats和两个CPU模块完成。
 * 当前状态：这是演示程序，尚不读取tensor.bin，也不解析quant.cfg或调用CUDA。
 */
#include "formats.h"
#include "metrics.h"
#include "mxfp8.h"
#include "nvfp4.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
// 阅读入口时先看main里的样例和调用顺序，再进入两个demo函数。
// 本文件负责演示与展示；真正的量化算法在mxfp8_cpu.cpp/nvfp4_cpu.cpp中。

// 用途：把存储编码转换成便于观察比特的十六进制文本。
// 输入：一个字节；输出：类似"0x79"的字符串；仅供日志/CSV展示。
std::string hex_byte(std::uint8_t code) {
  // uint8_t经常被输出流当作字符，转unsigned后才按整数显示十六进制。
  // 此转换只影响显示方式，不改变编码；setw(2)让0显示为0x00。
  std::ostringstream out;
  out << "0x" << std::uppercase << std::hex << std::setw(2)
      << std::setfill('0') << static_cast<unsigned>(code);
  return out.str();
}

// 用途：为各演示数据建立统一的CSV输出方式，避免各文件数字格式不同。
// 输入：文件路径；输出：已打开且配置完成的文件流，调用者负责写入内容。
std::ofstream open_csv(const fs::path& path) {
  // 文件打开/写入失败时抛异常，由main统一报告；classic保证CSV用小数点。
  // double的max_digits10保留足够数字，避免导出的指标或FP32输入被显示舍入。
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path);
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  return out;
}

// 用途：保留可查看的原始输入，使读者能对照量化前后结果。
// 输入：输出流、格式标签、输入数组；输出：写入CSV数据行，无返回值。
void write_inputs(std::ostream& out, const char* format,
                  const std::vector<float>& input) {
  for (std::size_t i = 0; i < input.size(); ++i) {
    out << format << ',' << i << ',' << input[i] << '\n';
  }
}

// 用途：把一次量化实验的误差与存储开销汇总成中文日志和CSV记录。
// 输入：原始/恢复数组、格式标签、实际量化负载字节数；输出：统计记录。
// 主流程关系：在quantize和dequantize之后调用，内部使用独立metrics模块。
void report_metrics(std::ostream& csv, const char* format,
                    const std::vector<float>& input,
                    const std::vector<float>& restored,
                    std::size_t payload_bytes) {
  // 输入是原始值、已恢复值及实际负载字节数；此处不重新进行量化。
  const auto errors = luxorion::calculate_metrics(input, restored);
  // 本演示的实际输入类型是 FP32；负载含 scale 和填充，不含文件 header。
  const auto compression = luxorion::calculate_compression(
      input.size() * sizeof(float), payload_bytes);
  csv << format << ',' << input.size() << ',' << errors.max_abs << ','
      << errors.mae << ',' << errors.mse << ',' << compression.original_bytes
      << ',' << payload_bytes << ',' << compression.compression_ratio << '\n';
  std::cout << format << " 误差与存储统计：\n"
            << "  最大绝对误差（MaxAbs）=" << errors.max_abs << '\n'
            << "  平均绝对误差（MAE）=" << errors.mae << '\n'
            << "  均方误差（MSE）=" << errors.mse << '\n'
            << "  原始 FP32 数据=" << compression.original_bytes << " 字节\n"
            << "  量化负载（包含缩放系数和填充）=" << payload_bytes << " 字节\n"
            << "  压缩率（原始字节数 / 量化负载字节数）="
            << compression.compression_ratio << '\n';
}

// 用途：运行一次MXFP8完整CPU实验，并展示每个元素用了什么scale和编码。
// 输入：FP32样例、CSV目录、指标输出流；输出：逐元素CSV、终端日志和指标。
// 主流程关系：由main调用，组织两个CPU接口，再将原始/恢复数组交给指标函数。
void demo_mxfp8(const fs::path& directory, const std::vector<float>& input,
                std::ostream& metrics) {
  // 核心流程只有两次调用：生成编码与scale，再用保存的结果恢复FP32数组。
  // 后续循环仅解释结果并导出CSV，额外decode是为了展示各列含义。
  const auto q = luxorion::mxfp8_quantize_cpu(input);
  const auto restored = luxorion::mxfp8_dequantize_cpu(q);
  auto csv = open_csv(directory / "mxfp8.csv");
  csv << "index,block,input,scale_code,scale,data_code,decoded,restored,abs_error\n";
  std::cout << "\nMXFP8 CPU 量化与反量化：每 32 个元素一块，尾块不补齐。\n"
            << "元素下标 | 块编号 | 原始值 | 缩放系数编码 | 缩放系数（scale） | 元素编码 | 还原值\n";
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::size_t block = i / 32;
    const float scale = luxorion::decode_e8m0(q.scales[block]);
    const float decoded = luxorion::decode_e4m3(q.data[i]);
    const double error = std::fabs(static_cast<double>(input[i]) - restored[i]);
    csv << i << ',' << block << ',' << input[i] << ',' << hex_byte(q.scales[block])
        << ',' << scale << ',' << hex_byte(q.data[i]) << ',' << decoded << ','
        << restored[i] << ',' << error << '\n';
    std::cout << i << " | " << block << " | " << input[i] << " | "
              << hex_byte(q.scales[block]) << " | " << scale << " | "
              << hex_byte(q.data[i]) << " | " << restored[i] << '\n';
  }
  csv.close();
  report_metrics(metrics, "mxfp8", input, restored, q.data.size() + q.scales.size());
}

// 用途：运行一次NVFP4完整CPU实验，额外展示两级scale和包含填充的字节布局。
// 输入：FP32样例、CSV目录、指标输出流；输出：元素/packed CSV、日志和指标。
// 主流程关系：由main调用；有效元素与存储字节分开展示，帮助检查尾块处理。
void demo_nvfp4(const fs::path& directory, const std::vector<float>& input,
                std::ostream& metrics) {
  // q保存element_count、packed_data、block_scales和global_scale。
  // restored只含有效元素；下面单独导出packed字节以观察补零存储。
  const auto q = luxorion::nvfp4_quantize_cpu(input);
  const auto restored = luxorion::nvfp4_dequantize_cpu(q);
  auto csv = open_csv(directory / "nvfp4.csv");
  csv << "index,block,input,global_scale,scale_code,block_scale,effective_scale,"
         "byte_index,nibble,packed_byte,data_code,decoded,restored,abs_error\n";
  std::cout << "\nNVFP4 CPU 量化与反量化：每 16 个元素一块，每块存储 8 字节，尾块补零。\n"
            << "全局缩放系数（global_scale）=" << q.global_scale << '\n'
            << "有效缩放系数 = 全局缩放系数 × 块缩放系数；偶数下标在高四位，奇数下标在低四位。\n"
            << "元素下标 | 块编号 | 原始值 | 块缩放系数编码 | 块缩放系数 | 有效缩放系数 | 字节下标 | 高低四位 | 打包字节 | 元素编码 | 还原值\n";
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::size_t block = i / 16;
    const std::size_t byte = i / 2;
    std::uint8_t first = 0, second = 0;
    luxorion::unpack_e2m1(q.packed_data[byte], first, second);
    // 偶数 index 在高四位，奇数 index 在低四位。
    const auto code = i % 2 == 0 ? first : second;
    const char* nibble = i % 2 == 0 ? "high" : "low";
    const float scale = luxorion::decode_e4m3(q.block_scales[block]);
    const float effective_scale = q.global_scale * scale;
    const float decoded = luxorion::decode_e2m1(code);
    const double error = std::fabs(static_cast<double>(input[i]) - restored[i]);
    csv << i << ',' << block << ',' << input[i] << ',' << q.global_scale << ','
        << hex_byte(q.block_scales[block]) << ',' << scale << ',' << effective_scale
        << ',' << byte << ',' << nibble << ',' << hex_byte(q.packed_data[byte])
        << ',' << hex_byte(code) << ',' << decoded << ',' << restored[i] << ',' << error << '\n';
    std::cout << i << " | " << block << " | " << input[i] << " | "
              << hex_byte(q.block_scales[block]) << " | " << scale << " | "
              << effective_scale << " | " << byte << " | "
              << (i % 2 == 0 ? "高四位" : "低四位") << " | "
              << hex_byte(q.packed_data[byte]) << " | " << hex_byte(code)
              << " | " << restored[i] << '\n';
  }
  csv.close();

  // 单独展示全部存储字节；有效元素的 CSV 不包含填充元素。
  auto packed = open_csv(directory / "nvfp4_packed.csv");
  packed << "byte_index,block,packed_byte,high_code,low_code,high_valid,low_valid\n";
  for (std::size_t byte = 0; byte < q.packed_data.size(); ++byte) {
    std::uint8_t first = 0, second = 0;
    luxorion::unpack_e2m1(q.packed_data[byte], first, second);
    packed << byte << ',' << byte / 8 << ',' << hex_byte(q.packed_data[byte])
           << ',' << hex_byte(first) << ',' << hex_byte(second) << ','
           << (byte * 2 < q.element_count) << ',' << (byte * 2 + 1 < q.element_count) << '\n';
  }
  packed.close();
  std::cout << "有效元素=" << q.element_count << " 个，存储容量="
            << q.packed_data.size() * 2 << " 个元素，填充="
            << q.packed_data.size() * 2 - q.element_count
            << " 个元素（编码均为零，详见 nvfp4_packed.csv）。\n";
  report_metrics(metrics, "nvfp4", input, restored,
                 q.packed_data.size() + q.block_scales.size() + sizeof(q.global_scale));
}
}  // namespace

// 用途：整个演示的控制入口，准备输入、创建输出目录并依次运行两个格式。
// 输入：可选CSV目录参数；输出：演示文件/日志，返回0成功、1失败。
// 阅读顺序：先看这里理解模块连接，再读两个demo，最后进入具体量化函数。
int main(int argc, char** argv) {
#ifdef _WIN32
  // Windows 控制台按 UTF-8 显示中文；重定向时日志仍输出 UTF-8 字节。
  SetConsoleOutputCP(CP_UTF8);
#endif
  try {
    if (argc > 2) throw std::invalid_argument("用法：luxorion [CSV 输出目录]");
    const fs::path directory = argc == 2 ? fs::path(argv[1]) : fs::path("output/cpu_demo");
    fs::create_directories(directory);
    std::cout << std::setprecision(9);
    std::cout << "Luxorion CPU 固定样例演示\n"
              << "下标与块编号从 0 开始；0x 开头表示十六进制编码。\n";

    // 固定教学样例；其余未赋值的位置均为零。
    // vector<float>(N,0)创建N个真正的零值，不是未初始化空间。
    // MXFP8长度33展示跨块scale，NVFP4长度19展示奇数尾元素和完整块填充。
    std::vector<float> mx_input(33, 0.0F);
    mx_input[0] = 1.0F;
    mx_input[1] = -1.1F;
    mx_input[2] = 15.5F;
    mx_input[3] = 260.0F;
    mx_input[32] = 600.0F;
    std::vector<float> nv_input(19, 0.0F);
    nv_input[0] = 2688.0F;
    nv_input[16] = 6.0F;
    nv_input[17] = -3.0F;
    nv_input[18] = 2.5F;

    auto inputs = open_csv(directory / "input.csv");
    // CSV输入是本程序导出的样例记录，不是从input/tensor.bin读取的数据。
    inputs << "format,index,input\n";
    write_inputs(inputs, "mxfp8", mx_input);
    write_inputs(inputs, "nvfp4", nv_input);
    inputs.close();
    auto metrics = open_csv(directory / "metrics.csv");
    metrics << "format,element_count,max_abs,mae,mse,original_bytes,payload_bytes,compression_ratio\n";
    demo_mxfp8(directory, mx_input, metrics);
    // 两个格式使用各自教学样例，目的是检查各自的机制；这些指标不能
    // 用来直接比较哪种格式更准。做精度对比时需给两种格式同一份输入。
    demo_nvfp4(directory, nv_input, metrics);
    metrics.close();
    std::cout << "\n演示完成，CSV 输出目录：" << fs::absolute(directory).u8string() << '\n'
              << "  input.csv：原始输入\n"
              << "  mxfp8.csv：MXFP8 编码、缩放系数、还原值和误差\n"
              << "  nvfp4.csv：NVFP4 编码、两级缩放系数、还原值和误差\n"
              << "  nvfp4_packed.csv：全部打包字节及填充位置\n"
              << "  metrics.csv：误差和压缩率汇总\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU 演示失败：" << error.what() << '\n';
    return 1;
  }
}
