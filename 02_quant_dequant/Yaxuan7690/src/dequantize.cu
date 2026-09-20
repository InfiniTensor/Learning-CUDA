// =============================================================================
//  dequantize.cu -- 低精度解包与反量化(GPU)
//
//  反量化公式:
//    MXFP8 / MXFP4 : x = e4m3/e5m2/e2m1(code) * 2^(scale_code - 127)
//    NVFP4         : x = e2m1(code) * e4m3(block_scale_code) / global_scale
//  最后由 f32_to_fmt<FmtF16/FmtBF16> 完成到目标输出类型的舍入,保证与 CPU 参考
//  实现逐位一致.
//
//  打包布局:4bit 格式每个字节存 2 个元素(下标偶数的元素在低半字节).
// =============================================================================
#include <algorithm>

#include "lpq/common.hpp"
#include "lpq/pipeline.hpp"
#include "lpq/vec_io.cuh"

namespace lpq {
namespace {

constexpr int kThreads = 256;

unsigned int grid_for_launch(int64_t work, int threads) {
    int64_t g = (work + threads - 1) / threads;
    if (g < 1)
        g = 1;
    if (g > 2147483647LL)
        g = 2147483647LL;
    return (unsigned int)g;
}

// -----------------------------------------------------------------------------
// MXFP8 / MXFP4 反量化(两者只差元素格式与打包位宽)
// -----------------------------------------------------------------------------
template <class OutT, class EFmt>
__global__ void k_dequant_mxfp8(const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, int64_t numel, int64_t block_size, int shift,
    int64_t num_blocks, OutT* __restrict__ out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nvec = numel >> 2;
    if (t < nvec) {
        const int64_t i0 = t << 2;
        // 4 个连续元素几乎总落在同一个缩放块内, 因此只需要 2 次块号计算:
        // 相等就查一次表。块号一律走 block_index(2 的幂时是移位), 避免原来的
        // same_block() 里那次 i64 整数除法 -- 那是这个 kernel 的主要 ALU 开销.
        const int64_t b0 = (num_blocks == 1) ? 0 : block_index(i0, block_size, shift);
        const int64_t b3 = (num_blocks == 1) ? 0 : block_index(i0 + 3, block_size, shift);
        float s[4];
        if (b0 == b3) {
            const float v = e8m0_to_f32(scales[b0]);
            s[0] = s[1] = s[2] = s[3] = v;
        } else {
            s[0] = e8m0_to_f32(scales[b0]);
            s[1] = e8m0_to_f32(scales[block_index(i0 + 1, block_size, shift)]);
            s[2] = e8m0_to_f32(scales[block_index(i0 + 2, block_size, shift)]);
            s[3] = e8m0_to_f32(scales[b3]);
        }
        float dec[4];
        if constexpr (ElemTraits<EFmt>::bits == 4) {
            // 4bit 元素: 4 个元素 = 2 字节, 按 uint16 读(按 uint32 读会越界)
            fmt_x4_4bit_to_f32<EFmt>(*reinterpret_cast<const uint16_t*>(packed + (i0 >> 1)), dec);
        } else {
            // 4 个 f8 编码一次性译码(sm_89+ 走硬件)
            fmt_x4_to_f32<EFmt>(*reinterpret_cast<const uint32_t*>(packed + i0), dec);
        }
        float v[4];
        for (int k = 0; k < 4; ++k) {
            v[k] = dec[k] * s[k];
        }
        OutTraits<OutT>::store4(out + i0, v);
    }
    if (t == 0) {
        for (int64_t i = nvec * 4; i < numel; ++i) {
            const int64_t b = (num_blocks == 1) ? 0 : block_index(i, block_size, shift);
            const float s = e8m0_to_f32(scales[b]);
            // 4bit 元素在字节里的位置: 偶下标低半字节, 奇下标高半字节
            const uint32_t code =
                (ElemTraits<EFmt>::bits == 4)
                    ? (uint32_t)((packed[i >> 1] >> (4 * (uint32_t)(i & 1))) & 0xFu)
                    : (uint32_t)packed[i];
            const float v = fmt_to_f32<EFmt>(code) * s;
            out[i] = OutTraits<OutT>::from_float(v);  // 尾部最多 3 个元素
        }
    }
}

// -----------------------------------------------------------------------------
// NVFP4 反量化
// -----------------------------------------------------------------------------
template <class OutT>
__global__ void k_dequant_nvfp4(const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ gscale, int64_t numel,
    int64_t block_size, int shift, int64_t num_blocks, OutT* __restrict__ out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nvec = numel >> 2;
    const float gf = gscale[1];  // 1 / global_scale
    if (t < nvec) {
        const int64_t i0 = t << 2;
        const uint16_t p = *reinterpret_cast<const uint16_t*>(packed + (i0 >> 1));
        const int64_t b0 = (num_blocks == 1) ? 0 : block_index(i0, block_size, shift);
        const int64_t b3 = (num_blocks == 1) ? 0 : block_index(i0 + 3, block_size, shift);
        float bs[4];
        if (b0 == b3) {
            const float v = fmt_to_f32<FmtE4M3>(scales[b0]) * gf;
            bs[0] = bs[1] = bs[2] = bs[3] = v;
        } else {
            bs[0] = fmt_to_f32<FmtE4M3>(scales[b0]) * gf;
            bs[1] = fmt_to_f32<FmtE4M3>(scales[block_index(i0 + 1, block_size, shift)]) * gf;
            bs[2] = fmt_to_f32<FmtE4M3>(scales[block_index(i0 + 2, block_size, shift)]) * gf;
            bs[3] = fmt_to_f32<FmtE4M3>(scales[b3]) * gf;
        }
        float v[4];
        for (int k = 0; k < 4; ++k) {
            v[k] = fmt_to_f32<FmtE2M1>((p >> (4 * k)) & 0xFu) * bs[k];
        }
        OutTraits<OutT>::store4(out + i0, v);
    }
    if (t == 0) {
        for (int64_t i = nvec * 4; i < numel; ++i) {
            const int64_t b = (num_blocks == 1) ? 0 : block_index(i, block_size, shift);
            const float s = fmt_to_f32<FmtE4M3>(scales[b]) * gf;
            const uint8_t byte = packed[i >> 1];
            const uint32_t code = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
            out[i] = OutTraits<OutT>::from_float(fmt_to_f32<FmtE2M1>(code) * s);
        }
    }
}

// -----------------------------------------------------------------------------
// 类型转换 kernel(fp16/bf16/fp32 -> fp32),测试与日志使用
// -----------------------------------------------------------------------------
template <class SrcT>
__global__ void k_to_float(const SrcT* __restrict__ src, float* __restrict__ dst, int64_t numel) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        dst[i] = OutTraits<SrcT>::to_float(src[i]);
    }
}

template <class SrcT>
__global__ void k_from_float(const float* __restrict__ src, SrcT* __restrict__ dst, int64_t numel) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
        dst[i] = OutTraits<SrcT>::from_float(src[i]);
    }
}

// -----------------------------------------------------------------------------
// 启动封装
// -----------------------------------------------------------------------------
template <class OutT>
void launch_dequant(
    const QuantizedDevice& q, const QuantConfig& cfg, OutT* d_out, cudaStream_t stream) {
    const int64_t nvec = (q.numel + 3) / 4;
    const unsigned int grid = grid_for_launch(nvec, kThreads);
    const int shift = pow2_shift(q.block_size);
    if (cfg.format == QuantFormat::MxFP4) {
        k_dequant_mxfp8<OutT, FmtE2M1><<<grid, kThreads, 0, stream>>>(q.packed.as<uint8_t>(),
            q.scales.as<uint8_t>(), q.numel, q.block_size, shift, q.num_blocks, d_out);
    } else if (cfg.format == QuantFormat::MxFP8) {
        if (cfg.mxfp8_e5m2) {
            k_dequant_mxfp8<OutT, FmtE5M2><<<grid, kThreads, 0, stream>>>(q.packed.as<uint8_t>(),
                q.scales.as<uint8_t>(), q.numel, q.block_size, shift, q.num_blocks, d_out);
        } else {
            k_dequant_mxfp8<OutT, FmtE4M3><<<grid, kThreads, 0, stream>>>(q.packed.as<uint8_t>(),
                q.scales.as<uint8_t>(), q.numel, q.block_size, shift, q.num_blocks, d_out);
        }
    } else {
        k_dequant_nvfp4<OutT><<<grid, kThreads, 0, stream>>>(q.packed.as<uint8_t>(),
            q.scales.as<uint8_t>(), q.global_scale.as<float>(), q.numel, q.block_size, shift,
            q.num_blocks, d_out);
    }
    LPQ_CUDA_CHECK_KERNEL();
}

template <class SrcT>
void launch_to_float(const SrcT* src, float* dst, int64_t numel, cudaStream_t stream) {
    const int64_t want = (numel + kThreads - 1) / kThreads;
    const unsigned int grid = grid_for_launch(std::min<int64_t>(want, 4096) * kThreads, kThreads);
    k_to_float<SrcT><<<grid, kThreads, 0, stream>>>(src, dst, numel);
    LPQ_CUDA_CHECK_KERNEL();
}

}  // namespace

double dequantize_on_device(
    const QuantizedDevice& q, const QuantConfig& cfg, void* d_output, cudaStream_t stream) {
    DeviceTimer timer;
    timer.start();
    switch (cfg.output_type) {
        case OutputType::FP16:
            launch_dequant<__half>(q, cfg, static_cast<__half*>(d_output), stream);
            break;
        case OutputType::BF16:
            launch_dequant<__nv_bfloat16>(q, cfg, static_cast<__nv_bfloat16*>(d_output), stream);
            break;
        default:
            launch_dequant<float>(q, cfg, static_cast<float*>(d_output), stream);
            break;
    }
    return timer.stop_ms();
}

double dequantize_on_device(const QuantizedDevice& q, const QuantConfig& cfg, void* d_output) {
    return dequantize_on_device(q, cfg, d_output, nullptr);
}

void convert_to_float_device(
    const void* src, InputType type, float* dst, int64_t numel, cudaStream_t stream) {
    if (type == InputType::FP32) {
        launch_to_float<float>(static_cast<const float*>(src), dst, numel, stream);
    } else {
        launch_to_float<__half>(static_cast<const __half*>(src), dst, numel, stream);
    }
}

void convert_to_float_device(
    const void* src, OutputType type, float* dst, int64_t numel, cudaStream_t stream) {
    switch (type) {
        case OutputType::FP16:
            launch_to_float<__half>(static_cast<const __half*>(src), dst, numel, stream);
            break;
        case OutputType::BF16:
            launch_to_float<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(src), dst, numel, stream);
            break;
        default:
            launch_to_float<float>(static_cast<const float*>(src), dst, numel, stream);
            break;
    }
}

}  // namespace lpq