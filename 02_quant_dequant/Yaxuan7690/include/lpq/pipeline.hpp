// =============================================================================
//  pipeline.hpp -- GPU 量化 / 反量化流水线
//
//  流水线阶段(与报告中的性能拆解一一对应):
//      1) amax   : 计算每个缩放块(或整个张量)的最大绝对值
//      2) scale  : 由 amax 生成缩放因子
//                  (MXFP8 / MXFP4 -> E8M0;NVFP4 -> E4M3 块缩放 + FP32 全局缩放)
//      3) quant  : 元素量化 + 打包写出
//      4) dequant: 解包 + 反量化到目标输出类型
//      5) metrics: 误差统计(max/MAE/MSE)
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "lpq/config.hpp"
#include "lpq/lowp_tensor.hpp"

namespace lpq {

// -----------------------------------------------------------------------------
// 设备缓冲区 RAII 封装
// -----------------------------------------------------------------------------
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;

    void reset();
    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    size_t size() const { return bytes_; }
    template <class T>
    T* as() {
        return reinterpret_cast<T*>(ptr_);
    }
    template <class T>
    const T* as() const {
        return reinterpret_cast<const T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;  // 逻辑大小(调用方请求的字节数)
    size_t cap_ = 0;    // 实际块大小; 走显存池复用时可能大于 bytes_
};

// -----------------------------------------------------------------------------
// 各阶段耗时(GPU 事件计时,单位 ms)
// -----------------------------------------------------------------------------
struct StageTimings {
    double amax_ms = 0.0;
    double scale_ms = 0.0;
    double quant_ms = 0.0;
    double dequant_ms = 0.0;
    double metrics_ms = 0.0;
};

// -----------------------------------------------------------------------------
// 量化结果(全部驻留显存)
// -----------------------------------------------------------------------------
struct QuantizedDevice {
    DeviceBuffer packed;        // 打包后的低精度元素
    DeviceBuffer scales;        // 缩放因子(E8M0 / E4M3 编码)
    DeviceBuffer global_scale;  // NVFP4 的全张量 FP32 缩放(长度 1;MX 格式不用)
    int64_t numel = 0;
    int64_t block_size = 0;  // 每个缩放因子覆盖的元素数
    int64_t num_blocks = 0;
    float global_scale_value = 1.0f;
    float global_scale_inv = 1.0f;
    // 量化阶段从显存读取输入张量的次数. MX 格式的 block 模式使用融合 kernel,
    // 输入只需读一遍(=1);NVFP4 必须先算出全张量 amax 才能定出全局缩放, 因此固定为
    // 两遍(=2);NVFP4 的 4/6 自适应块缩放还要多一遍只读扫描来挑块缩放因子(=3).
    // 这个数字用于把「有效带宽」换算成真实的显存流量.
    int input_passes = 2;
    // 误差统计是否已经融进量化 kernel 完成(见 quantize_on_device 的 out_stats 参数).
    // 为 false 时调用方需要自己再跑一遍 compute_error_stats.
    bool stats_fused = false;
    StageTimings timings;
};

// 元素编码类型(与 LowpHeader::elem_kind 对应)
enum class ElemKind : uint32_t { E4M3 = 0, E5M2 = 1, E2M1 = 2 };

ElemKind elem_kind_of(const QuantConfig& cfg);

// 在设备上执行量化流水线.d_input 为 fp32 或 fp16 位模式,长度 numel.
QuantizedDevice quantize_on_device(const void* d_input, InputType in_type, int64_t numel,
    const QuantConfig& cfg, cudaStream_t stream);

// 反量化:把 q 还原成 output_type 指定的元素类型,写入 d_output(长度 numel).
double dequantize_on_device(
    const QuantizedDevice& q, const QuantConfig& cfg, void* d_output, cudaStream_t stream);

// -----------------------------------------------------------------------------
// 误差统计
// -----------------------------------------------------------------------------
struct ErrorStats {
    double max_abs = 0.0;
    double mae = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    double rel_mse = 0.0;  // mse / mean(ref^2)
    // 带符号平均误差 mean(dequant - original). RNE 通常非零(有偏), SR 期望为 0.
    double mean_signed_error = 0.0;
    double ref_amax = 0.0;
    int64_t numel = 0;
};

// 误差统计的累加策略:
//   Precise -- 逐元素转 fp64 再累加, 与 CPU 双精度参考逐位可比(参考路径);
//   Fast    -- 线程内 (hi, lo) 补偿 fp32 累加, 线程间仍在 fp64 归约.
//              避开消费级 GPU 上 1/64 吞吐的 fp64 ALU 瓶颈,
//              相对误差 ~1e-15, 正常没有理由取 Precise.
enum class AccumMode { Precise, Fast };

// 同上, 但顺带算出误差统计.
//
// 误差统计需要 4 个全局和 + 一个 max, 传统做法是事后单跑一个 metrics kernel, 把
// fp32 原始输入和反量化结果各读一遍(8192^2 下是 402MB, 比量化本身还重). 但量化
// kernel 手里**本来就有**原始值 x 和刚算出的量化码, 顺手反量化一次就能累加, 于是这
// 一整遍访存可以直接省掉 -- 统计变成零额外显存流量.
//
// 代价是量化 kernel 里多出约 50 条/元素的 fp32 指令; 实测仍然停在访存带宽上.
// 返回的 QuantizedDevice::stats_fused 指示统计是否真的被融合完成(当前所有量化路径
// 都支持, 但保留这个标志以便将来新增路径时能安全回退到独立 kernel).
QuantizedDevice quantize_on_device(const void* d_input, InputType in_type, int64_t numel,
    const QuantConfig& cfg, cudaStream_t stream, ErrorStats* out_stats,
    AccumMode mode = AccumMode::Fast);

// d_ref: 原始输入(fp32/fp16),d_test: 反量化结果(fp16/bf16/fp32)
ErrorStats compute_error_stats(const void* d_ref, InputType ref_type, const void* d_test,
    OutputType test_type, int64_t numel, cudaStream_t stream, AccumMode mode = AccumMode::Fast);

// -----------------------------------------------------------------------------
//  融合统计用的设备端归约目标: 4 x fp64 累加器 + 1 x fp32 最大值.
//  是一小块进程内缓存的显存(只有 36 字节, 但每次 cudaMalloc/cudaFree 在 kernel
//  本身只有 1ms 时能占到 4%). 调用前需要由调用方清零.
// -----------------------------------------------------------------------------
struct StatsReduction {
    double* acc = nullptr;
    float* max = nullptr;
};
StatsReduction stats_reduction_scratch();

// 把 (4 个全局和, max|d|, 元素个数) 组装成 ErrorStats. 独立 metrics kernel 与
// 量化 kernel 里融合的统计共用这一份收尾逻辑, 保证两条路径给出同一组数值.
ErrorStats finalize_error_stats(const double acc[4], float max_abs, int64_t numel);

// fp16/bf16/fp32 设备缓冲区 -> float 缓冲区(用于日志与比对;测试也会用到)
void convert_to_float_device(
    const void* src, InputType type, float* dst, int64_t numel, cudaStream_t stream);
void convert_to_float_device(
    const void* src, OutputType type, float* dst, int64_t numel, cudaStream_t stream);

// 便捷封装:在默认流上创建/销毁
QuantizedDevice quantize_on_device(
    const void* d_input, InputType in_type, int64_t numel, const QuantConfig& cfg);
double dequantize_on_device(const QuantizedDevice& q, const QuantConfig& cfg, void* d_output);

}  // namespace lpq