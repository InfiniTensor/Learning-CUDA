#include "mxfp8.cuh"

#include <cmath>
#include <cstdint>

namespace mxfp8 {
    namespace {
        constexpr unsigned kFullMask = 0xffffffffu;
        constexpr int kThreads = 256;

        template <typename T>
        struct InputTraits;

        template <>
        struct InputTraits<float> {
            __device__ __forceinline__ static float to_float(float x) {
                return x;
            }
        };

        template <>
        struct InputTraits<__half> {
            __device__ __forceinline__ static float to_float(float x) {
                return __half2float(x);
            }
        };

        __device__ __forceinline__ BF16 float_to_bf16_rne(float x) {
            uint32_t bits = __float_as_uint(x);
            const uint32_t exp = (bits >> 23) & 0xffu;
            const uint32_t frac = bits & 0x7fffffu;

            BF16 out{};

            // Inf / NaN。
            if (exp == 0xffu) {
                out.bits = static_cast<uint16_t>(bits >> 16);
                if (frac != 0) {
                    out.bits |= 0x0040u;  // 静默 NaN。
                }
                return out;
            }

            const uint32_t retained_lsb = (bits >> 16) & 1u;
            bits += 0x7fffu + retained_lsb;
            out.bits = static_cast<uint16_t>(bits >> 16);
            return out;
        }

        template <typename T>
        struct OutputTraits;

        template <>
        struct OutputTraits<float> {
            __device__ __forceinline__ static float from_float(float x) {
                return x;
            }
        };

        template <>
        struct OutputTraits<__half> {
            __device__ __forceinline__ static __half from_float(float x) {
                return __float2half_rn(x);
            }
        };

        template <>
        struct OutputTraits<BF16> {
            __device__ __forceinline__ static BF16 from_float(float x) {
                return float_to_bf16_rne(x);
            }
        };

        __device__ __forceinline__ uint32_t round_shift_rne(uint32_t value, int shift) {
            if (shift <= 0) {
                return value;
            }

            if (shift >= 32) {
                return 0;
            }

            uint32_t kept = value >> shift;
            const uint32_t remainder = value & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1);

            if (remainder > halfway ||
                    (remainder == halfway && (kept & 1u))) {
                ++kept;
            }

            return kept;
        }

        __device__ __forceinline__ float warp_reduce_max(float value) {
            value = fmaxf(value, __shfl_down_sync(kFullMask, value, 16));
            value = fmaxf(value, __shfl_down_sync(kFullMask, value, 8));
            value = fmaxf(value, __shfl_down_sync(kFullMask, value, 4));
            value = fmaxf(value, __shfl_down_sync(kFullMask, value, 2));
            value = fmaxf(value, __shfl_down_sync(kFullMask, value, 1));

            return value;
        }

        // 根据 amax 计算 2^k 缩放指数。
        __device__ __forceinline__ int scale_exp_from_amax(float amax) {
            if (!(amax > 0.0f)) {
                return 0;  // 全零或全 NaN block。
            }

            if (isinf(amax)) {
                return 127;
            }

            const uint32_t bits = __float_as_uint(amax);
            const uint32_t e32 = (bits >> 23) & 0xffu;
            const uint32_t frac32 = bits & 0x7fffffu;

            int exponent;
            uint32_t significand;

            if (e32 != 0) {
                exponent = static_cast<int>(e32) - 127;
                significand = (1u << 23) | frac32;
            } else {
                // 规格化 FP32 次正规数。
                const int shift = __clz(frac32) - 8;
                significand = frac32 << shift;
                exponent = -126 - shift;
            }

            // E4M3 最大有限值为 448。
            constexpr uint32_t kE4M3MaxSignificand = 0xE00000u;
            int k = exponent - 8 + (significand > kE4M3MaxSignificand);

            if (k < -127) {
                k = -127;
            } else if (k > 127) {
                k = 127;
            }

            return k;
        }

        __device__ __forceinline__ uint8_t encode_e8m0(int k) {
            if (k < -127) {
                k = -127;
            } else if (k > 127) {
                k = 127;
            }

            return static_cast<uint8_t>(k + 127);
        }

        __device__ __forceinline__ int decode_e8m0(uint8_t code) {
            return static_cast<int>(code) - 127;
        }

        // 直接将 x / 2^k 编码为 E4M3。
        __device__ __forceinline__ uint8_t float_to_e4m3_scaled(float x, int k) {
            const uint32_t bits = __float_as_uint(x);

            const uint32_t sign = (bits >> 31) & 1u;
            const uint32_t e32 = (bits >> 23) & 0xffu;
            const uint32_t frac32 = bits & 0x7fffffu;

            // 有符号零。
            if (e32 == 0 && frac32 == 0) {
                return static_cast<uint8_t>(sign << 7);
            }

            // FP32 Inf / NaN。
            if (e32 == 0xffu) {
                if (frac32 != 0) {
                    // E4M3 NaN。
                    return static_cast<uint8_t>((sign << 7) | 0x7fu);
                }

                // E4M3 无 Inf，饱和到 ±448。
                return static_cast<uint8_t>((sign << 7) | 0x7eu);
            }

            uint32_t significand;
            int e4;

            if (e32 != 0) {
                significand = (1u << 23) | frac32;
                // 计算 E4M3 指数字段。
                e4 = static_cast<int>(e32) - k - 120;
            } else {
                // 规格化 FP32 次正规数。
                const int shift = __clz(frac32) - 8;
                significand = frac32 << shift;
                const int exponent = -126 - shift;
                e4 = exponent - k + 7;
            }

            uint32_t exp_field;
            uint32_t frac_field;

            if (e4 >= 1) {
                // 保留 3 位尾数并执行 RNE。
                uint32_t rounded = round_shift_rne(significand, 20);

                // 处理舍入进位。
                if (rounded == 16) {
                    rounded = 8;
                    ++e4;
                }

                exp_field = static_cast<uint32_t>(e4);
                frac_field = rounded & 0x7u;

                // 超出范围时饱和到 448。
                if (exp_field > 15 ||
                        (exp_field == 15 && frac_field == 7)) {
                    exp_field = 15;
                    frac_field = 6;
                }
            } else {
                // 转换为 E4M3 次正规数。
                const int shift = 21 - e4;
                const uint32_t q = round_shift_rne(significand, shift);

                if (q == 0) {
                    exp_field = 0;
                    frac_field = 0;
                } else if (q >= 8) {
                    // 舍入到最小正规数。
                    exp_field = 1;
                    frac_field = 0;
                } else {
                    exp_field = 0;
                    frac_field = q;
                }
            }

            return static_cast<uint8_t>(
                (sign << 7) | (exp_field << 3) | frac_field);
        }

        __device__ __forceinline__ float e4m3_to_float(uint8_t q) {
            const uint32_t sign = (q >> 7) & 1u;
            const uint32_t exp = (q >> 3) & 0xfu;
            const uint32_t frac = q & 0x7u;

            float value;

            if (exp == 0) {
                if (frac == 0) {
                    value = 0.0f;
                } else {
                    value = ldexpf(static_cast<float>(frac) / 8.0f, -6);
                }
            } else if (exp == 0xfu && frac == 0x7u) {
                value = NAN;
            } else {
                const float significand = 1.0f + static_cast<float>(frac) / 8.0f;
                value = ldexpf(significand, static_cast<int>(exp) - 7);
            }

            return sign ? -value : value;
        }

        template <typename InputT>
        __global__ void quantize_kernel_v1(const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int lane = threadIdx.x & 31;
            const int mx_block = tid >> 5;
            const bool valid = tid < n;

            // 尾部 lane 用 0 参与归约。
            const float x = valid ? InputTraits<InputT>::to_float(input[tid]) : 0.0f;
            const float amax = warp_reduce_max(fabsf(x));

            int k = 0;
            if (lane == 0 && valid) {
                k = scale_exp_from_amax(amax);
                scales[mx_block] = encode_e8m0(k);
            }

            k = __shfl_sync(kFullMask, k, 0);

            if (valid) {
                quantized[tid] = float_to_e4m3_scaled(x, k);
            }
        }

        template <typename InputT>
        __global__ void quantize_kernel_v2(const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {
            constexpr int kItemsPerWarp = 2;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            const int num_mx_blocks = (n + 31) >> 5;

            const int mx_block0 = global_warp * kItemsPerWarp;
            const int mx_block1 = mx_block0 + 1;

            const int tid0 = mx_block0 * 32 + lane;
            const int tid1 = mx_block1 * 32 + lane;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;

            // 两个 MXFP8 block 独立加载。
            const float x0 =
                valid0 ? InputTraits<InputT>::to_float(input[tid0]) : 0.0f;

            const float x1 =
                valid1 ? InputTraits<InputT>::to_float(input[tid1]) : 0.0f;

            float amax0 = fabsf(x0);
            float amax1 = fabsf(x1);

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other0 =
                    __shfl_down_sync(kFullMask, amax0, offset);

                const float other1 =
                    __shfl_down_sync(kFullMask, amax1, offset);

                amax0 = fmaxf(amax0, other0);
                amax1 = fmaxf(amax1, other1);
            }

            int k0 = 0;
            int k1 = 0;

            if (lane == 0) {
                if (mx_block0 < num_mx_blocks) {
                    k0 = scale_exp_from_amax(amax0);
                    scales[mx_block0] = encode_e8m0(k0);
                }

                if (mx_block1 < num_mx_blocks) {
                    k1 = scale_exp_from_amax(amax1);
                    scales[mx_block1] = encode_e8m0(k1);
                }
            }

            k0 = __shfl_sync(kFullMask, k0, 0);
            k1 = __shfl_sync(kFullMask, k1, 0);

            if (valid0) {
                quantized[tid0] =
                    float_to_e4m3_scaled(x0, k0);
            }

            if (valid1) {
                quantized[tid1] =
                    float_to_e4m3_scaled(x1, k1);
            }
        }

        template <typename InputT>
        __global__ void quantize_kernel_v3(const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {
            constexpr int kMxBlocksPerWarp = 4;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            const int num_mx_blocks = (n + kBlockSize - 1) / kBlockSize;

            // 一个 warp 处理连续 4 个 MXFP8 block。
            const int mx_block0 = global_warp * kMxBlocksPerWarp + 0;
            const int mx_block1 = global_warp * kMxBlocksPerWarp + 1;
            const int mx_block2 = global_warp * kMxBlocksPerWarp + 2;
            const int mx_block3 = global_warp * kMxBlocksPerWarp + 3;

            const int tid0 = mx_block0 * kBlockSize + lane;
            const int tid1 = mx_block1 * kBlockSize + lane;
            const int tid2 = mx_block2 * kBlockSize + lane;
            const int tid3 = mx_block3 * kBlockSize + lane;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;
            const bool valid2 = tid2 < n;
            const bool valid3 = tid3 < n;

            // 读取 4 个 MXFP8 block。

            const float x0 =
                valid0 ? InputTraits<InputT>::to_float(input[tid0]) : 0.0f;

            const float x1 =
                valid1 ? InputTraits<InputT>::to_float(input[tid1]) : 0.0f;

            const float x2 =
                valid2 ? InputTraits<InputT>::to_float(input[tid2]) : 0.0f;

            const float x3 =
                valid3 ? InputTraits<InputT>::to_float(input[tid3]) : 0.0f;

            // 4 条独立的 warp 归约链。
            float amax0 = fabsf(x0);
            float amax1 = fabsf(x1);
            float amax2 = fabsf(x2);
            float amax3 = fabsf(x3);

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other0 =
                    __shfl_down_sync(kFullMask, amax0, offset);

                const float other1 =
                    __shfl_down_sync(kFullMask, amax1, offset);

                const float other2 =
                    __shfl_down_sync(kFullMask, amax2, offset);

                const float other3 =
                    __shfl_down_sync(kFullMask, amax3, offset);

                amax0 = fmaxf(amax0, other0);
                amax1 = fmaxf(amax1, other1);
                amax2 = fmaxf(amax2, other2);
                amax3 = fmaxf(amax3, other3);
            }

            int k0 = 0;
            int k1 = 0;
            int k2 = 0;
            int k3 = 0;

            // lane 0 计算各 block 的 scale。
            if (lane == 0) {
                if (mx_block0 < num_mx_blocks) {
                    k0 = scale_exp_from_amax(amax0);
                    scales[mx_block0] = encode_e8m0(k0);
                }

                if (mx_block1 < num_mx_blocks) {
                    k1 = scale_exp_from_amax(amax1);
                    scales[mx_block1] = encode_e8m0(k1);
                }

                if (mx_block2 < num_mx_blocks) {
                    k2 = scale_exp_from_amax(amax2);
                    scales[mx_block2] = encode_e8m0(k2);
                }

                if (mx_block3 < num_mx_blocks) {
                    k3 = scale_exp_from_amax(amax3);
                    scales[mx_block3] = encode_e8m0(k3);
                }
            }

            // 广播 4 个 scale exponent。
            k0 = __shfl_sync(kFullMask, k0, 0);
            k1 = __shfl_sync(kFullMask, k1, 0);
            k2 = __shfl_sync(kFullMask, k2, 0);
            k3 = __shfl_sync(kFullMask, k3, 0);

            // E4M3 编码。

            if (valid0) {
                quantized[tid0] =
                    float_to_e4m3_scaled(x0, k0);
            }

            if (valid1) {
                quantized[tid1] =
                    float_to_e4m3_scaled(x1, k1);
            }

            if (valid2) {
                quantized[tid2] =
                    float_to_e4m3_scaled(x2, k2);
            }

            if (valid3) {
                quantized[tid3] =
                    float_to_e4m3_scaled(x3, k3);
            }
        }

        template <typename InputT>
        __global__ void quantize_kernel_v4(const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {
            constexpr int kMxBlocksPerWarp = 8;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            // 一个 warp 处理连续 8 个 MXFP8 block。

            const int mx_block0 = global_warp * kMxBlocksPerWarp + 0;
            const int mx_block1 = global_warp * kMxBlocksPerWarp + 1;
            const int mx_block2 = global_warp * kMxBlocksPerWarp + 2;
            const int mx_block3 = global_warp * kMxBlocksPerWarp + 3;
            const int mx_block4 = global_warp * kMxBlocksPerWarp + 4;
            const int mx_block5 = global_warp * kMxBlocksPerWarp + 5;
            const int mx_block6 = global_warp * kMxBlocksPerWarp + 6;
            const int mx_block7 = global_warp * kMxBlocksPerWarp + 7;

            const int tid0 = mx_block0 * kBlockSize + lane;
            const int tid1 = mx_block1 * kBlockSize + lane;
            const int tid2 = mx_block2 * kBlockSize + lane;
            const int tid3 = mx_block3 * kBlockSize + lane;
            const int tid4 = mx_block4 * kBlockSize + lane;
            const int tid5 = mx_block5 * kBlockSize + lane;
            const int tid6 = mx_block6 * kBlockSize + lane;
            const int tid7 = mx_block7 * kBlockSize + lane;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;
            const bool valid2 = tid2 < n;
            const bool valid3 = tid3 < n;
            const bool valid4 = tid4 < n;
            const bool valid5 = tid5 < n;
            const bool valid6 = tid6 < n;
            const bool valid7 = tid7 < n;

            // 读取 8 个 MXFP8 block。

            const float x0 =
                valid0 ? InputTraits<InputT>::to_float(input[tid0]) : 0.0f;

            const float x1 =
                valid1 ? InputTraits<InputT>::to_float(input[tid1]) : 0.0f;

            const float x2 =
                valid2 ? InputTraits<InputT>::to_float(input[tid2]) : 0.0f;

            const float x3 =
                valid3 ? InputTraits<InputT>::to_float(input[tid3]) : 0.0f;

            const float x4 =
                valid4 ? InputTraits<InputT>::to_float(input[tid4]) : 0.0f;

            const float x5 =
                valid5 ? InputTraits<InputT>::to_float(input[tid5]) : 0.0f;

            const float x6 =
                valid6 ? InputTraits<InputT>::to_float(input[tid6]) : 0.0f;

            const float x7 =
                valid7 ? InputTraits<InputT>::to_float(input[tid7]) : 0.0f;

            // 8 条独立的 warp 归约链。

            float amax0 = fabsf(x0);
            float amax1 = fabsf(x1);
            float amax2 = fabsf(x2);
            float amax3 = fabsf(x3);
            float amax4 = fabsf(x4);
            float amax5 = fabsf(x5);
            float amax6 = fabsf(x6);
            float amax7 = fabsf(x7);

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other0 =
                    __shfl_down_sync(kFullMask, amax0, offset);
                const float other1 =
                    __shfl_down_sync(kFullMask, amax1, offset);
                const float other2 =
                    __shfl_down_sync(kFullMask, amax2, offset);
                const float other3 =
                    __shfl_down_sync(kFullMask, amax3, offset);
                const float other4 =
                    __shfl_down_sync(kFullMask, amax4, offset);
                const float other5 =
                    __shfl_down_sync(kFullMask, amax5, offset);
                const float other6 =
                    __shfl_down_sync(kFullMask, amax6, offset);
                const float other7 =
                    __shfl_down_sync(kFullMask, amax7, offset);

                amax0 = fmaxf(amax0, other0);
                amax1 = fmaxf(amax1, other1);
                amax2 = fmaxf(amax2, other2);
                amax3 = fmaxf(amax3, other3);
                amax4 = fmaxf(amax4, other4);
                amax5 = fmaxf(amax5, other5);
                amax6 = fmaxf(amax6, other6);
                amax7 = fmaxf(amax7, other7);
            }

            int k0 = 0;
            int k1 = 0;
            int k2 = 0;
            int k3 = 0;
            int k4 = 0;
            int k5 = 0;
            int k6 = 0;
            int k7 = 0;

            // lane 0 计算 8 个 scale。

            if (lane == 0) {
                if (mx_block0 < num_mx_blocks) {
                    k0 = scale_exp_from_amax(amax0);
                    scales[mx_block0] = encode_e8m0(k0);
                }

                if (mx_block1 < num_mx_blocks) {
                    k1 = scale_exp_from_amax(amax1);
                    scales[mx_block1] = encode_e8m0(k1);
                }

                if (mx_block2 < num_mx_blocks) {
                    k2 = scale_exp_from_amax(amax2);
                    scales[mx_block2] = encode_e8m0(k2);
                }

                if (mx_block3 < num_mx_blocks) {
                    k3 = scale_exp_from_amax(amax3);
                    scales[mx_block3] = encode_e8m0(k3);
                }

                if (mx_block4 < num_mx_blocks) {
                    k4 = scale_exp_from_amax(amax4);
                    scales[mx_block4] = encode_e8m0(k4);
                }

                if (mx_block5 < num_mx_blocks) {
                    k5 = scale_exp_from_amax(amax5);
                    scales[mx_block5] = encode_e8m0(k5);
                }

                if (mx_block6 < num_mx_blocks) {
                    k6 = scale_exp_from_amax(amax6);
                    scales[mx_block6] = encode_e8m0(k6);
                }

                if (mx_block7 < num_mx_blocks) {
                    k7 = scale_exp_from_amax(amax7);
                    scales[mx_block7] = encode_e8m0(k7);
                }
            }

            // 广播 8 个 scale exponent。

            k0 = __shfl_sync(kFullMask, k0, 0);
            k1 = __shfl_sync(kFullMask, k1, 0);
            k2 = __shfl_sync(kFullMask, k2, 0);
            k3 = __shfl_sync(kFullMask, k3, 0);
            k4 = __shfl_sync(kFullMask, k4, 0);
            k5 = __shfl_sync(kFullMask, k5, 0);
            k6 = __shfl_sync(kFullMask, k6, 0);
            k7 = __shfl_sync(kFullMask, k7, 0);

            // E4M3 编码。

            if (valid0) {
                quantized[tid0] =
                    float_to_e4m3_scaled(x0, k0);
            }

            if (valid1) {
                quantized[tid1] =
                    float_to_e4m3_scaled(x1, k1);
            }

            if (valid2) {
                quantized[tid2] =
                    float_to_e4m3_scaled(x2, k2);
            }

            if (valid3) {
                quantized[tid3] =
                    float_to_e4m3_scaled(x3, k3);
            }

            if (valid4) {
                quantized[tid4] =
                    float_to_e4m3_scaled(x4, k4);
            }

            if (valid5) {
                quantized[tid5] =
                    float_to_e4m3_scaled(x5, k5);
            }

            if (valid6) {
                quantized[tid6] =
                    float_to_e4m3_scaled(x6, k6);
            }

            if (valid7) {
                quantized[tid7] =
                    float_to_e4m3_scaled(x7, k7);
            }
        }

        template <typename InputT>
        __global__ void quantize_kernel_v4_packed(
            const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {

            constexpr int kMxBlocksPerWarp = 8;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            const int mx_block0 = global_warp * kMxBlocksPerWarp + 0;
            const int mx_block1 = global_warp * kMxBlocksPerWarp + 1;
            const int mx_block2 = global_warp * kMxBlocksPerWarp + 2;
            const int mx_block3 = global_warp * kMxBlocksPerWarp + 3;
            const int mx_block4 = global_warp * kMxBlocksPerWarp + 4;
            const int mx_block5 = global_warp * kMxBlocksPerWarp + 5;
            const int mx_block6 = global_warp * kMxBlocksPerWarp + 6;
            const int mx_block7 = global_warp * kMxBlocksPerWarp + 7;

            const int tid0 = mx_block0 * kBlockSize + lane;
            const int tid1 = mx_block1 * kBlockSize + lane;
            const int tid2 = mx_block2 * kBlockSize + lane;
            const int tid3 = mx_block3 * kBlockSize + lane;
            const int tid4 = mx_block4 * kBlockSize + lane;
            const int tid5 = mx_block5 * kBlockSize + lane;
            const int tid6 = mx_block6 * kBlockSize + lane;
            const int tid7 = mx_block7 * kBlockSize + lane;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;
            const bool valid2 = tid2 < n;
            const bool valid3 = tid3 < n;
            const bool valid4 = tid4 < n;
            const bool valid5 = tid5 < n;
            const bool valid6 = tid6 < n;
            const bool valid7 = tid7 < n;

            // 读取 8 个 MXFP8 block。

            const float x0 =
                valid0 ? InputTraits<InputT>::to_float(input[tid0]) : 0.0f;
            const float x1 =
                valid1 ? InputTraits<InputT>::to_float(input[tid1]) : 0.0f;
            const float x2 =
                valid2 ? InputTraits<InputT>::to_float(input[tid2]) : 0.0f;
            const float x3 =
                valid3 ? InputTraits<InputT>::to_float(input[tid3]) : 0.0f;
            const float x4 =
                valid4 ? InputTraits<InputT>::to_float(input[tid4]) : 0.0f;
            const float x5 =
                valid5 ? InputTraits<InputT>::to_float(input[tid5]) : 0.0f;
            const float x6 =
                valid6 ? InputTraits<InputT>::to_float(input[tid6]) : 0.0f;
            const float x7 =
                valid7 ? InputTraits<InputT>::to_float(input[tid7]) : 0.0f;

            // 8 条独立的 warp 归约链。

            float amax0 = fabsf(x0);
            float amax1 = fabsf(x1);
            float amax2 = fabsf(x2);
            float amax3 = fabsf(x3);
            float amax4 = fabsf(x4);
            float amax5 = fabsf(x5);
            float amax6 = fabsf(x6);
            float amax7 = fabsf(x7);

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other0 =
                    __shfl_down_sync(kFullMask, amax0, offset);
                const float other1 =
                    __shfl_down_sync(kFullMask, amax1, offset);
                const float other2 =
                    __shfl_down_sync(kFullMask, amax2, offset);
                const float other3 =
                    __shfl_down_sync(kFullMask, amax3, offset);
                const float other4 =
                    __shfl_down_sync(kFullMask, amax4, offset);
                const float other5 =
                    __shfl_down_sync(kFullMask, amax5, offset);
                const float other6 =
                    __shfl_down_sync(kFullMask, amax6, offset);
                const float other7 =
                    __shfl_down_sync(kFullMask, amax7, offset);

                amax0 = fmaxf(amax0, other0);
                amax1 = fmaxf(amax1, other1);
                amax2 = fmaxf(amax2, other2);
                amax3 = fmaxf(amax3, other3);
                amax4 = fmaxf(amax4, other4);
                amax5 = fmaxf(amax5, other5);
                amax6 = fmaxf(amax6, other6);
                amax7 = fmaxf(amax7, other7);
            }

            int k0 = 0;
            int k1 = 0;
            int k2 = 0;
            int k3 = 0;
            int k4 = 0;
            int k5 = 0;
            int k6 = 0;
            int k7 = 0;

            // 计算并打包写回 8 个 scale。

            if (lane == 0) {

                // 完整 8 个 block 路径。
                if (mx_block7 < num_mx_blocks) {

                    k0 = scale_exp_from_amax(amax0);
                    k1 = scale_exp_from_amax(amax1);
                    k2 = scale_exp_from_amax(amax2);
                    k3 = scale_exp_from_amax(amax3);
                    k4 = scale_exp_from_amax(amax4);
                    k5 = scale_exp_from_amax(amax5);
                    k6 = scale_exp_from_amax(amax6);
                    k7 = scale_exp_from_amax(amax7);

                    const uint8_t s0 = encode_e8m0(k0);
                    const uint8_t s1 = encode_e8m0(k1);
                    const uint8_t s2 = encode_e8m0(k2);
                    const uint8_t s3 = encode_e8m0(k3);
                    const uint8_t s4 = encode_e8m0(k4);
                    const uint8_t s5 = encode_e8m0(k5);
                    const uint8_t s6 = encode_e8m0(k6);
                    const uint8_t s7 = encode_e8m0(k7);

                    const uint64_t packed =
                        static_cast<uint64_t>(s0)
                        | (static_cast<uint64_t>(s1) << 8)
                        | (static_cast<uint64_t>(s2) << 16)
                        | (static_cast<uint64_t>(s3) << 24)
                        | (static_cast<uint64_t>(s4) << 32)
                        | (static_cast<uint64_t>(s5) << 40)
                        | (static_cast<uint64_t>(s6) << 48)
                        | (static_cast<uint64_t>(s7) << 56);

                    // 起始位置按 8 字节对齐。
                    *reinterpret_cast<uint64_t*>(scales + mx_block0) =
                        packed;
                }

                // 尾部 block 路径。
                else {
                    if (mx_block0 < num_mx_blocks) {
                        k0 = scale_exp_from_amax(amax0);
                        scales[mx_block0] = encode_e8m0(k0);
                    }

                    if (mx_block1 < num_mx_blocks) {
                        k1 = scale_exp_from_amax(amax1);
                        scales[mx_block1] = encode_e8m0(k1);
                    }

                    if (mx_block2 < num_mx_blocks) {
                        k2 = scale_exp_from_amax(amax2);
                        scales[mx_block2] = encode_e8m0(k2);
                    }

                    if (mx_block3 < num_mx_blocks) {
                        k3 = scale_exp_from_amax(amax3);
                        scales[mx_block3] = encode_e8m0(k3);
                    }

                    if (mx_block4 < num_mx_blocks) {
                        k4 = scale_exp_from_amax(amax4);
                        scales[mx_block4] = encode_e8m0(k4);
                    }

                    if (mx_block5 < num_mx_blocks) {
                        k5 = scale_exp_from_amax(amax5);
                        scales[mx_block5] = encode_e8m0(k5);
                    }

                    if (mx_block6 < num_mx_blocks) {
                        k6 = scale_exp_from_amax(amax6);
                        scales[mx_block6] = encode_e8m0(k6);
                    }

                    if (mx_block7 < num_mx_blocks) {
                        k7 = scale_exp_from_amax(amax7);
                        scales[mx_block7] = encode_e8m0(k7);
                    }
                }
            }

            // 广播 8 个 scale exponent。

            k0 = __shfl_sync(kFullMask, k0, 0);
            k1 = __shfl_sync(kFullMask, k1, 0);
            k2 = __shfl_sync(kFullMask, k2, 0);
            k3 = __shfl_sync(kFullMask, k3, 0);
            k4 = __shfl_sync(kFullMask, k4, 0);
            k5 = __shfl_sync(kFullMask, k5, 0);
            k6 = __shfl_sync(kFullMask, k6, 0);
            k7 = __shfl_sync(kFullMask, k7, 0);

            // E4M3 编码。

            if (valid0) {
                quantized[tid0] =
                    float_to_e4m3_scaled(x0, k0);
            }

            if (valid1) {
                quantized[tid1] =
                    float_to_e4m3_scaled(x1, k1);
            }

            if (valid2) {
                quantized[tid2] =
                    float_to_e4m3_scaled(x2, k2);
            }

            if (valid3) {
                quantized[tid3] =
                    float_to_e4m3_scaled(x3, k3);
            }

            if (valid4) {
                quantized[tid4] =
                    float_to_e4m3_scaled(x4, k4);
            }

            if (valid5) {
                quantized[tid5] =
                    float_to_e4m3_scaled(x5, k5);
            }

            if (valid6) {
                quantized[tid6] =
                    float_to_e4m3_scaled(x6, k6);
            }

            if (valid7) {
                quantized[tid7] =
                    float_to_e4m3_scaled(x7, k7);
            }
        }

        template <typename InputT>
        __global__ void quantize_kernel_v5(
            const InputT* __restrict__ input,
            uint8_t* __restrict__ quantized,
            uint8_t* __restrict__ scales,
            int n) {

            constexpr int kMxBlocksPerWarp = 8;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            const int base_mx_block =
                global_warp * kMxBlocksPerWarp;

            const int mx_block0 = base_mx_block + 0;
            const int mx_block1 = base_mx_block + 1;
            const int mx_block2 = base_mx_block + 2;
            const int mx_block3 = base_mx_block + 3;
            const int mx_block4 = base_mx_block + 4;
            const int mx_block5 = base_mx_block + 5;
            const int mx_block6 = base_mx_block + 6;
            const int mx_block7 = base_mx_block + 7;

            const int tid0 = mx_block0 * kBlockSize + lane;
            const int tid1 = mx_block1 * kBlockSize + lane;
            const int tid2 = mx_block2 * kBlockSize + lane;
            const int tid3 = mx_block3 * kBlockSize + lane;
            const int tid4 = mx_block4 * kBlockSize + lane;
            const int tid5 = mx_block5 * kBlockSize + lane;
            const int tid6 = mx_block6 * kBlockSize + lane;
            const int tid7 = mx_block7 * kBlockSize + lane;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;
            const bool valid2 = tid2 < n;
            const bool valid3 = tid3 < n;
            const bool valid4 = tid4 < n;
            const bool valid5 = tid5 < n;
            const bool valid6 = tid6 < n;
            const bool valid7 = tid7 < n;

            // 读取 8 个 MXFP8 block。

            const float x0 =
                valid0 ? InputTraits<InputT>::to_float(input[tid0]) : 0.0f;
            const float x1 =
                valid1 ? InputTraits<InputT>::to_float(input[tid1]) : 0.0f;
            const float x2 =
                valid2 ? InputTraits<InputT>::to_float(input[tid2]) : 0.0f;
            const float x3 =
                valid3 ? InputTraits<InputT>::to_float(input[tid3]) : 0.0f;
            const float x4 =
                valid4 ? InputTraits<InputT>::to_float(input[tid4]) : 0.0f;
            const float x5 =
                valid5 ? InputTraits<InputT>::to_float(input[tid5]) : 0.0f;
            const float x6 =
                valid6 ? InputTraits<InputT>::to_float(input[tid6]) : 0.0f;
            const float x7 =
                valid7 ? InputTraits<InputT>::to_float(input[tid7]) : 0.0f;

            float amax0 = fabsf(x0);
            float amax1 = fabsf(x1);
            float amax2 = fabsf(x2);
            float amax3 = fabsf(x3);
            float amax4 = fabsf(x4);
            float amax5 = fabsf(x5);
            float amax6 = fabsf(x6);
            float amax7 = fabsf(x7);

            // 蝶形归约，使各 lane 都得到完整 amax。

#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax0 = fmaxf(
                    amax0,
                    __shfl_xor_sync(kFullMask, amax0, mask));

                amax1 = fmaxf(
                    amax1,
                    __shfl_xor_sync(kFullMask, amax1, mask));

                amax2 = fmaxf(
                    amax2,
                    __shfl_xor_sync(kFullMask, amax2, mask));

                amax3 = fmaxf(
                    amax3,
                    __shfl_xor_sync(kFullMask, amax3, mask));

                amax4 = fmaxf(
                    amax4,
                    __shfl_xor_sync(kFullMask, amax4, mask));

                amax5 = fmaxf(
                    amax5,
                    __shfl_xor_sync(kFullMask, amax5, mask));

                amax6 = fmaxf(
                    amax6,
                    __shfl_xor_sync(kFullMask, amax6, mask));

                amax7 = fmaxf(
                    amax7,
                    __shfl_xor_sync(kFullMask, amax7, mask));
            }

            // lanes 0~7 并行计算 8 个 scale。

            float my_amax = 0.0f;

            if (lane == 0) {
                my_amax = amax0;
            } else if (lane == 1) {
                my_amax = amax1;
            } else if (lane == 2) {
                my_amax = amax2;
            } else if (lane == 3) {
                my_amax = amax3;
            } else if (lane == 4) {
                my_amax = amax4;
            } else if (lane == 5) {
                my_amax = amax5;
            } else if (lane == 6) {
                my_amax = amax6;
            } else if (lane == 7) {
                my_amax = amax7;
            }

            int k_local = 0;

            const int my_mx_block = base_mx_block + lane;

            if (lane < kMxBlocksPerWarp &&
                    my_mx_block < num_mx_blocks) {
                k_local = scale_exp_from_amax(my_amax);
            }

            // 广播 8 个 scale exponent。

            const int k0 = __shfl_sync(kFullMask, k_local, 0);
            const int k1 = __shfl_sync(kFullMask, k_local, 1);
            const int k2 = __shfl_sync(kFullMask, k_local, 2);
            const int k3 = __shfl_sync(kFullMask, k_local, 3);
            const int k4 = __shfl_sync(kFullMask, k_local, 4);
            const int k5 = __shfl_sync(kFullMask, k_local, 5);
            const int k6 = __shfl_sync(kFullMask, k_local, 6);
            const int k7 = __shfl_sync(kFullMask, k_local, 7);

            // 打包写回 8 个 scale。

            if (lane == 0) {

                if (mx_block7 < num_mx_blocks) {

                    const uint8_t s0 = encode_e8m0(k0);
                    const uint8_t s1 = encode_e8m0(k1);
                    const uint8_t s2 = encode_e8m0(k2);
                    const uint8_t s3 = encode_e8m0(k3);
                    const uint8_t s4 = encode_e8m0(k4);
                    const uint8_t s5 = encode_e8m0(k5);
                    const uint8_t s6 = encode_e8m0(k6);
                    const uint8_t s7 = encode_e8m0(k7);

                    const uint64_t packed =
                        static_cast<uint64_t>(s0)
                        | (static_cast<uint64_t>(s1) << 8)
                        | (static_cast<uint64_t>(s2) << 16)
                        | (static_cast<uint64_t>(s3) << 24)
                        | (static_cast<uint64_t>(s4) << 32)
                        | (static_cast<uint64_t>(s5) << 40)
                        | (static_cast<uint64_t>(s6) << 48)
                        | (static_cast<uint64_t>(s7) << 56);

                    *reinterpret_cast<uint64_t*>(
                        scales + base_mx_block) = packed;
                }

                // 尾部 block 路径。
                else {
                    if (mx_block0 < num_mx_blocks)
                        scales[mx_block0] = encode_e8m0(k0);

                    if (mx_block1 < num_mx_blocks)
                        scales[mx_block1] = encode_e8m0(k1);

                    if (mx_block2 < num_mx_blocks)
                        scales[mx_block2] = encode_e8m0(k2);

                    if (mx_block3 < num_mx_blocks)
                        scales[mx_block3] = encode_e8m0(k3);

                    if (mx_block4 < num_mx_blocks)
                        scales[mx_block4] = encode_e8m0(k4);

                    if (mx_block5 < num_mx_blocks)
                        scales[mx_block5] = encode_e8m0(k5);

                    if (mx_block6 < num_mx_blocks)
                        scales[mx_block6] = encode_e8m0(k6);

                    if (mx_block7 < num_mx_blocks)
                        scales[mx_block7] = encode_e8m0(k7);
                }
            }

            // E4M3 编码。

            if (valid0) {
                quantized[tid0] =
                    float_to_e4m3_scaled(x0, k0);
            }

            if (valid1) {
                quantized[tid1] =
                    float_to_e4m3_scaled(x1, k1);
            }

            if (valid2) {
                quantized[tid2] =
                    float_to_e4m3_scaled(x2, k2);
            }

            if (valid3) {
                quantized[tid3] =
                    float_to_e4m3_scaled(x3, k3);
            }

            if (valid4) {
                quantized[tid4] =
                    float_to_e4m3_scaled(x4, k4);
            }

            if (valid5) {
                quantized[tid5] =
                    float_to_e4m3_scaled(x5, k5);
            }

            if (valid6) {
                quantized[tid6] =
                    float_to_e4m3_scaled(x6, k6);
            }

            if (valid7) {
                quantized[tid7] =
                    float_to_e4m3_scaled(x7, k7);
            }
        }

        template <typename OutputT>
        __global__ void dequantize_kernel(const uint8_t* __restrict__ quantized,
            const uint8_t* __restrict__ scales,
            OutputT* __restrict__ output,
            int n) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n) {
                return;
            }

            const int mx_block = tid >> 5;
            const uint8_t scale_code = scales[mx_block];

            if (scale_code == 0xffu) {
                output[tid] = OutputTraits<OutputT>::from_float(NAN);
                return;
            }

            const int k = decode_e8m0(scale_code);
            const float decoded = e4m3_to_float(quantized[tid]);
            const float restored = ldexpf(decoded, k);

            output[tid] = OutputTraits<OutputT>::from_float(restored);
        }

        template <typename OutputT>
        __global__ void dequantize_kernel_v2(
            const uint8_t* __restrict__ quantized,
            const uint8_t* __restrict__ scales,
            OutputT* __restrict__ output,
            int n) {

            constexpr int kMxBlocksPerWarp = 8;

            const int lane = threadIdx.x & 31;
            const int warp_in_block = threadIdx.x >> 5;
            const int warps_per_block = blockDim.x >> 5;

            const int global_warp =
                blockIdx.x * warps_per_block + warp_in_block;

            // 一个 warp 处理连续 8 个 MXFP8 block。
            const int base_mx_block =
                global_warp * kMxBlocksPerWarp;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            // lanes 0~7 读取 8 个连续 scale。

            uint32_t my_scale = 0;

            if (lane < kMxBlocksPerWarp) {
                const int mx_block = base_mx_block + lane;

                if (mx_block < num_mx_blocks) {
                    my_scale = static_cast<uint32_t>(scales[mx_block]);
                }
            }

            // 将 8 个 scale 广播给整个 warp。

            const uint8_t s0 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 0));

            const uint8_t s1 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 1));

            const uint8_t s2 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 2));

            const uint8_t s3 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 3));

            const uint8_t s4 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 4));

            const uint8_t s5 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 5));

            const uint8_t s6 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 6));

            const uint8_t s7 = static_cast<uint8_t>(
                __shfl_sync(kFullMask, my_scale, 7));

            // 每个 lane 处理 8 个连续 block 中的对应元素。

            const int base_tid =
                base_mx_block * kBlockSize + lane;

            const int tid0 = base_tid;
            const int tid1 = base_tid + 32;
            const int tid2 = base_tid + 64;
            const int tid3 = base_tid + 96;
            const int tid4 = base_tid + 128;
            const int tid5 = base_tid + 160;
            const int tid6 = base_tid + 192;
            const int tid7 = base_tid + 224;

            const bool valid0 = tid0 < n;
            const bool valid1 = tid1 < n;
            const bool valid2 = tid2 < n;
            const bool valid3 = tid3 < n;
            const bool valid4 = tid4 < n;
            const bool valid5 = tid5 < n;
            const bool valid6 = tid6 < n;
            const bool valid7 = tid7 < n;

            // 读取 8 个量化值。

            const uint8_t q0 = valid0 ? quantized[tid0] : 0;
            const uint8_t q1 = valid1 ? quantized[tid1] : 0;
            const uint8_t q2 = valid2 ? quantized[tid2] : 0;
            const uint8_t q3 = valid3 ? quantized[tid3] : 0;
            const uint8_t q4 = valid4 ? quantized[tid4] : 0;
            const uint8_t q5 = valid5 ? quantized[tid5] : 0;
            const uint8_t q6 = valid6 ? quantized[tid6] : 0;
            const uint8_t q7 = valid7 ? quantized[tid7] : 0;

            // 8 条独立的反量化链。

            if (valid0) {
                if (s0 == 0xffu) {
                    output[tid0] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k0 = decode_e8m0(s0);
                    const float d0 = e4m3_to_float(q0);
                    const float r0 = ldexpf(d0, k0);

                    output[tid0] =
                        OutputTraits<OutputT>::from_float(r0);
                }
            }

            if (valid1) {
                if (s1 == 0xffu) {
                    output[tid1] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k1 = decode_e8m0(s1);
                    const float d1 = e4m3_to_float(q1);
                    const float r1 = ldexpf(d1, k1);

                    output[tid1] =
                        OutputTraits<OutputT>::from_float(r1);
                }
            }

            if (valid2) {
                if (s2 == 0xffu) {
                    output[tid2] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k2 = decode_e8m0(s2);
                    const float d2 = e4m3_to_float(q2);
                    const float r2 = ldexpf(d2, k2);

                    output[tid2] =
                        OutputTraits<OutputT>::from_float(r2);
                }
            }

            if (valid3) {
                if (s3 == 0xffu) {
                    output[tid3] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k3 = decode_e8m0(s3);
                    const float d3 = e4m3_to_float(q3);
                    const float r3 = ldexpf(d3, k3);

                    output[tid3] =
                        OutputTraits<OutputT>::from_float(r3);
                }
            }

            if (valid4) {
                if (s4 == 0xffu) {
                    output[tid4] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k4 = decode_e8m0(s4);
                    const float d4 = e4m3_to_float(q4);
                    const float r4 = ldexpf(d4, k4);

                    output[tid4] =
                        OutputTraits<OutputT>::from_float(r4);
                }
            }

            if (valid5) {
                if (s5 == 0xffu) {
                    output[tid5] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k5 = decode_e8m0(s5);
                    const float d5 = e4m3_to_float(q5);
                    const float r5 = ldexpf(d5, k5);

                    output[tid5] =
                        OutputTraits<OutputT>::from_float(r5);
                }
            }

            if (valid6) {
                if (s6 == 0xffu) {
                    output[tid6] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k6 = decode_e8m0(s6);
                    const float d6 = e4m3_to_float(q6);
                    const float r6 = ldexpf(d6, k6);

                    output[tid6] =
                        OutputTraits<OutputT>::from_float(r6);
                }
            }

            if (valid7) {
                if (s7 == 0xffu) {
                    output[tid7] =
                        OutputTraits<OutputT>::from_float(NAN);
                } else {
                    const int k7 = decode_e8m0(s7);
                    const float d7 = e4m3_to_float(q7);
                    const float r7 = ldexpf(d7, k7);

                    output[tid7] =
                        OutputTraits<OutputT>::from_float(r7);
                }
            }
        }

        template <typename InputT>
        void launch_quantize(const InputT* input,
            uint8_t* quantized,
            uint8_t* scales,
            int n,
            cudaStream_t stream) {
            if (n <= 0) {
                return;
            }

            constexpr int kMxBlocksPerWarp = 8;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            const int warps_needed =
                (num_mx_blocks + kMxBlocksPerWarp - 1)
            / kMxBlocksPerWarp;

            const int warps_per_cta = kThreads / 32;

            const int blocks =
                (warps_needed + warps_per_cta - 1)
            / warps_per_cta;

            quantize_kernel_v5<InputT>
            <<<blocks, kThreads, 0, stream>>>(
                input,
                quantized,
                scales,
                n);
        }

        template <typename OutputT>
        void launch_dequantize(const uint8_t* quantized,
            const uint8_t* scales,
            OutputT* output,
            int n,
            cudaStream_t stream) {
            if (n <= 0) {
                return;
            }

            constexpr int kMxBlocksPerWarp = 8;

            const int num_mx_blocks =
                (n + kBlockSize - 1) / kBlockSize;

            const int warps_needed =
                (num_mx_blocks + kMxBlocksPerWarp - 1)
            / kMxBlocksPerWarp;

            const int warps_per_cta =
                kThreads / 32;

            const int blocks =
                (warps_needed + warps_per_cta - 1)
            / warps_per_cta;

            dequantize_kernel_v2<OutputT>
            <<<blocks, kThreads, 0, stream>>>(
                quantized,
                scales,
                output,
                n);
        }
    }  // namespace

    void quantize(const float* input,
        uint8_t* quantized,
        uint8_t* scales,
        int n,
        cudaStream_t stream) {
        launch_quantize(input, quantized, scales, n, stream);
    }

    void quantize(const __half* input,
        uint8_t* quantized,
        uint8_t* scales,
        int n,
        cudaStream_t stream) {
        launch_quantize(input, quantized, scales, n, stream);
    }

    void dequantize(const uint8_t* quantized,
        const uint8_t* scales,
        float* output,
        int n,
        cudaStream_t stream) {
        launch_dequantize(quantized, scales, output, n, stream);
    }

    void dequantize(const uint8_t* quantized,
        const uint8_t* scales,
        __half* output,
        int n,
        cudaStream_t stream) {
        launch_dequantize(quantized, scales, output, n, stream);
    }

    void dequantize(const uint8_t* quantized,
        const uint8_t* scales,
        BF16* output,
        int n,
        cudaStream_t stream) {
        launch_dequantize(quantized, scales, output, n, stream);
    }

}  // namespace mxfp8
