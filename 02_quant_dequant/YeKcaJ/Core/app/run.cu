// 文件任务：组织输入、GPU 调用、CPU 校验和结果输出，不定义 kernel。
#include "app/commands.h"
#include "runtime/workspace.cuh"
#include "reference/oracle.h"
#include "reference/output_compare.cuh"
#include "io/tensor_io.h"
#include "io/output.cuh"
#include "common/timing.h"

namespace pipeline {

// 完整 run 流程的后半段：反量化到 T -> CPU 校验 -> 写张量 -> 输出 JSON。
// CPU 一致性检查与量化损失不同：后者比较原始输入和最终低精度舍入后的输出。
template <typename T>
void finish(Workspace& w, const Tensor& t, const Packed& q, const std::string& path, float quant_ms,
            double gpu_wall, bool quant_verified) {
  const float dequant_ms = w.n ? w.events.measure([&] { w.launch_dequant<T>(); }) : 0;
  const auto output = w.download_output<T>();

  // 验证实际 GPU 结果，同时保留“原始输入到目标 dtype”的量化损失统计。
  const auto expected = reference_dequantize(q);
  cuda_output::compare(output, expected);
  if (!fs::path(path).parent_path().empty())
    fs::create_directories(fs::path(path).parent_path());
  cuda_output::write(path, t.rows, t.cols, output);

  double max_abs = 0, mae = 0, mse = 0;
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    const double value = static_cast<float>(output[i]);
    if (!std::isfinite(value)) {
      ++nonfinite;
      continue;
    }
    const double diff = std::fabs(value - t.values[i]);
    max_abs = std::max(max_abs, diff);
    mae += diff;
    mse += diff * diff;
  }
  if (w.n) {
    mae /= w.n;
    mse /= w.n;
  }

  // payload 不计文件头，但包含 NVFP4 的全局 scale；file 压缩率计入双方文件头。
  const auto payload = q.data.size() + q.scales.size() + (nvfp4 ? 4 : 0);
  std::cout << "{\"format\":\"" << format_name << "\",\"input_dtype\":\"fp" << t.element_bytes * 8
            << "\",\"output_dtype\":\"" << cuda_output::Format<T>::name << "\",\"scale_mode\":\""
            << (q.options.tensor ? "tensor" : "block") << "\",\"rounding\":\""
            << (q.options.stochastic ? "stochastic" : "nearest") << "\",\"seed\":" << q.options.seed
            << ",\"elements\":" << w.n << ",\"nonfinite_output\":" << nonfinite;
  // FP16 输出可能溢出；此时不把有限子集的误差伪装成完整张量误差。
  if (nonfinite)
    std::cout << ",\"max_abs_error\":null,\"mae\":null,\"mse\":null";
  else
    std::cout << ",\"max_abs_error\":" << max_abs << ",\"mae\":" << mae << ",\"mse\":" << mse;
  std::cout << ",\"compression_payload\":"
            << (payload ? double(w.n * t.element_bytes) / payload : 0) << ",\"compression_file\":"
            << double(28 + w.n * t.element_bytes) / (72 + q.data.size() + q.scales.size())
            << ",\"quant_kernel_ms\":" << quant_ms
            << ",\"quant_setup_upload_compute_download_ms\":" << gpu_wall
            << ",\"dequant_kernel_ms\":" << dequant_ms << ",\"quant_logical_GBps\":"
            << (quant_ms > 0 ? double(4 * w.n + payload) / (quant_ms * 1e6) : 0)
            << ",\"dequant_logical_GBps\":"
            << (dequant_ms > 0 ? double(sizeof(T) * w.n + payload) / (dequant_ms * 1e6) : 0)
            << ",\"cpu_dequant_match\":true,\"cpu_quant_match\":"
            << (quant_verified ? "true" : "null") << "}\n";
}

// --dequant-file 路径：直接上传已有 packed 权重，不重复执行量化。
template <typename T>
void restore(const Packed& q, const std::string& path) {
  Workspace w(checked_count(q.rows, q.cols), q.options);
  w.upload(q);
  const float ms = w.n ? w.events.measure([&] { w.launch_dequant<T>(); }) : 0;
  auto output = w.download_output<T>();
  cuda_output::compare(output, reference_dequantize(q));
  if (!fs::path(path).parent_path().empty())
    fs::create_directories(fs::path(path).parent_path());
  cuda_output::write(path, q.rows, q.cols, output);
  std::cout << "dequant_kernel_ms=" << ms << '\n';
}

void restore_file(const std::string& input, const std::string& output, const std::string& dtype) {
  const auto q = read_packed(input);
  if (dtype == "fp32")
    restore<float>(q, output);
  else if (dtype == "fp16")
    restore<__half>(q, output);
  else if (dtype == "bf16")
    restore<__nv_bfloat16>(q, output);
  else
    throw std::runtime_error("invalid output dtype");
}

void run_pipeline(int argc, char** argv) {
  // 普通流程第一步：检查参数、固定 block 大小，并防止输入输出路径互相覆盖。
  if (argc != 10)
    throw std::runtime_error(
        "usage: pipeline input quantized output block|tensor nearest|stochastic fp32|fp16|bf16 block_size seed verify|noverify");
  const std::string mode = argv[4], rounding = argv[5], dtype = argv[6], verification = argv[9];
  if ((mode != "block" && mode != "tensor") ||
      (rounding != "nearest" && rounding != "stochastic") ||
      (dtype != "fp32" && dtype != "fp16" && dtype != "bf16") ||
      (verification != "verify" && verification != "noverify"))
    throw std::runtime_error("invalid pipeline option");
  const auto seed = std::stoull(argv[8]);
  if (seed > UINT32_MAX)
    throw std::runtime_error("seed exceeds uint32");
  Options o{mode == "tensor", rounding == "stochastic", std::stoull(argv[7]),
            static_cast<std::uint32_t>(seed)};
  validate(o);
  if (fs::absolute(argv[1]).lexically_normal() == fs::absolute(argv[2]).lexically_normal() ||
      fs::absolute(argv[1]).lexically_normal() == fs::absolute(argv[3]).lexically_normal() ||
      fs::absolute(argv[2]).lexically_normal() == fs::absolute(argv[3]).lexically_normal())
    throw std::runtime_error("input/output paths must differ");

  // 第二步：读取输入，再计时显存分配、上传、量化和下载。
  // wall 不包含读文件、CPU 校验、写文件或 Workspace 析构释放。
  const auto t = read_input(argv[1]);
  const auto start = Clock::now();
  Workspace w(t.values.size(), o);
  w.upload(t.values);
  const float quant_ms = w.n ? w.events.measure([&] { w.launch_quant(); }) : 0;
  auto q = w.download(t);
  const double wall = elapsed(start);

  // 第三步：按需校验 CPU 量化结果，并写入/回读 .lpq 检查文件往返一致性。
  if (verification == "verify")
    compare_packed(q, reference_quantize(t, o));
  write_packed(argv[2], q);
  compare_packed(q, read_packed(argv[2]));

  // 第四步：按输出 dtype 实例化反量化流程，保存张量并报告误差与性能。
  if (dtype == "fp16")
    finish<__half>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
  else if (dtype == "bf16")
    finish<__nv_bfloat16>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
  else
    finish<float>(w, t, q, argv[3], quant_ms, wall, verification == "verify");
}

}  // namespace pipeline
