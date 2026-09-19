#include "kernel_operator.h"

#ifndef HADAMARD_ASCEND_VECTOR_SCALE
#define HADAMARD_ASCEND_VECTOR_SCALE 0
#endif

// 本项目原生 Ascend C 实现；数学顺序沿用本项目 FWHT，不复制厂商 SDK 实现。
// 默认两方法共用标量scale；可选宏仅让VectorGather改用Muls，RNE与标量除法量化不变。
template<class T, bool Transform, bool Quantize> class HadamardKernel {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR packed_output,
                               GM_ADDR scale_output, uint64_t rows, uint32_t n,
                               float transform_scale, uint32_t method) {
        n_ = n;
        padded_n_ = (n + 7) / 8 * 8;  // 每个 FP32 bank 都从 32B 对齐地址开始。
        uint32_t stages = 0;
        for (uint32_t stride = 1; stride < n; stride *= 2) ++stages;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), rows * n);
        if constexpr (Quantize) {
            packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(packed_output), rows * ((n + 1) / 2));
            scales_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale_output), rows);
        } else {
            output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), rows * n);
        }

        pipe_.InitBuffer(input_queue_, 1, 256 * sizeof(T));
        pipe_.InitBuffer(typed_queue_, 1, 256 * sizeof(T));
        pipe_.InitBuffer(current_buffer_, padded_n_ * sizeof(float));
        pipe_.InitBuffer(next_buffer_, padded_n_ * sizeof(float));
        if constexpr (Transform) {
            if (method == 1) {
                pipe_.InitBuffer(peer_buffer_, padded_n_ * sizeof(float));
                pipe_.InitBuffer(combined_buffer_, 2 * padded_n_ * sizeof(float));
                const uint32_t index_bytes = stages ? stages * padded_n_ * sizeof(uint32_t) : 32;
                pipe_.InitBuffer(partner_offsets_buffer_, index_bytes);
                pipe_.InitBuffer(select_offsets_buffer_, index_bytes);
            }
        }
        if constexpr (Quantize) {
            pipe_.InitBuffer(packed_queue_, 1, 128);
            pipe_.InitBuffer(scale_queue_, 1, 32);
            pipe_.InitBuffer(divisor_buffer_, 32);
        }

        auto typed_input = input_queue_.template AllocTensor<T>();
        auto typed_output = typed_queue_.template AllocTensor<T>();
        auto current_base = current_buffer_.template Get<float>();
        auto next_base = next_buffer_.template Get<float>();
        for (uint32_t i = 0; i < padded_n_; ++i) {
            current_base.SetValue(i, 0.0f);
            next_base.SetValue(i, 0.0f);
        }
        for (uint32_t i = 0; i < 256; ++i) typed_output.SetValue(i, static_cast<T>(0));
        if constexpr (Transform) {
            if (method == 1) InitializeOffsets();
        }
        AscendC::LocalTensor<uint8_t> packed_local;
        AscendC::LocalTensor<float> scale_local;
        if constexpr (Quantize) {
            packed_local = packed_queue_.template AllocTensor<uint8_t>();
            scale_local = scale_queue_.template AllocTensor<float>();
            for (uint32_t i = 0; i < 128; ++i) packed_local.SetValue(i, uint8_t(0));
            for (uint32_t i = 0; i < 8; ++i) scale_local.SetValue(i, 0.0f);
            auto divisor = divisor_buffer_.template Get<float>();
            divisor.SetValue(0, 7.0f);
        }
        Fence<AscendC::HardEvent::S_V>();

        const uint32_t row_bytes = n * sizeof(T);
        const uint8_t right_pad = ((32 - row_bytes % 32) % 32) / sizeof(T);
        const AscendC::DataCopyExtParams typed_copy{1, row_bytes, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> input_pad{true, 0, right_pad, static_cast<T>(0)};
        for (uint64_t row = AscendC::GetBlockIdx(); row < rows; row += AscendC::GetBlockNum()) {
            auto current = current_base;
            auto next = next_base;
            AscendC::DataCopyPad(typed_input, input_[row * n], typed_copy, input_pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(current, typed_input, AscendC::RoundMode::CAST_NONE, n);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (Transform) {
                if (method == 1) VectorButterfly(current, next);
                #if HADAMARD_ASCEND_VECTOR_SCALE
                if (method == 1) {
                    // count重载原位逐元素相乘；不产生跨repeat的源/目标依赖。
                    AscendC::Muls(current, current, transform_scale, static_cast<int32_t>(n));
                    AscendC::PipeBarrier<PIPE_V>();
                } else
                #endif
                {
                    Fence<AscendC::HardEvent::V_S>();
                    if (method == 0) ScalarButterfly(current);
                    for (uint32_t i = 0; i < n; ++i)
                        current.SetValue(i, current.GetValue(i) * transform_scale);
                    Fence<AscendC::HardEvent::S_V>();
                }
                AscendC::Cast(typed_output, current, AscendC::RoundMode::CAST_RINT, n);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (Quantize) {
                    // 融合必须先舍入到公开 dtype，再读取这些值参与量化。
                    AscendC::Cast(current, typed_output, AscendC::RoundMode::CAST_NONE, n);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (Quantize) {
                Fence<AscendC::HardEvent::V_S>();
                QuantizeRow(current, packed_local, scale_local);
                Fence<AscendC::HardEvent::S_MTE3>();
                const AscendC::DataCopyExtParams byte_copy{1, (n + 1) / 2, 0, 0, 0};
                const AscendC::DataCopyExtParams scale_copy{1, uint32_t(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(packed_[row * ((n + 1) / 2)], packed_local, byte_copy);
                AscendC::DataCopyPad(scales_[row], scale_local, scale_copy);
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(output_[row * n], typed_output, typed_copy);
            }
            // 下一行可能复用 typed_output/packed/scale UB，必须等待前一行搬出完成。
            Fence<AscendC::HardEvent::MTE3_S>();
        }
        input_queue_.FreeTensor(typed_input);
        typed_queue_.FreeTensor(typed_output);
        if constexpr (Quantize) {
            packed_queue_.FreeTensor(packed_local);
            scale_queue_.FreeTensor(scale_local);
        }
    }

private:
    template<AscendC::HardEvent Event> __aicore__ inline void Fence() {
        auto id = pipe_.template AllocEventID<Event>();
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
        pipe_.template ReleaseEventID<Event>(id);
    }

    __aicore__ inline void InitializeOffsets() {
        auto partner = partner_offsets_buffer_.template Get<uint32_t>();
        auto select = select_offsets_buffer_.template Get<uint32_t>();
        auto peer = peer_buffer_.template Get<float>();
        auto combined = combined_buffer_.template Get<float>();
        for (uint32_t i = 0; i < padded_n_; ++i) {
            peer.SetValue(i, 0.0f);
            combined.SetValue(i, 0.0f);
            combined.SetValue(padded_n_ + i, 0.0f);
        }
        uint32_t stage = 0;
        for (uint32_t stride = 1; stride < n_; stride *= 2, ++stage) {
            for (uint32_t i = 0; i < padded_n_; ++i) {
                const uint32_t p = i < n_ ? (i ^ stride) * sizeof(float) : 0;
                const uint32_t s = i < n_ ? (i + ((i & stride) ? padded_n_ : 0)) * sizeof(float) : 0;
                partner.SetValue(stage * padded_n_ + i, p);
                select.SetValue(stage * padded_n_ + i, s);
            }
        }
    }

    __aicore__ inline void ScalarButterfly(AscendC::LocalTensor<float>& values) {
        for (uint32_t stride = 1; stride < n_; stride *= 2) {
            for (uint32_t start = 0; start < n_; start += 2 * stride) {
                for (uint32_t i = 0; i < stride; ++i) {
                    const float a = values.GetValue(start + i);
                    const float b = values.GetValue(start + i + stride);
                    values.SetValue(start + i, a + b);
                    values.SetValue(start + i + stride, a - b);
                }
            }
        }
    }

    __aicore__ inline void VectorButterfly(AscendC::LocalTensor<float>& values,
                                          AscendC::LocalTensor<float>& next) {
        auto peer = peer_buffer_.template Get<float>();
        auto combined = combined_buffer_.template Get<float>();
        auto partner = partner_offsets_buffer_.template Get<uint32_t>();
        auto select = select_offsets_buffer_.template Get<uint32_t>();
        uint32_t stage = 0;
        for (uint32_t stride = 1; stride < n_; stride *= 2, ++stage) {
            AscendC::Gather(peer, values, partner[stage * padded_n_], 0, n_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(combined, values, peer, n_);
            AscendC::Sub(combined[padded_n_], peer, values, n_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(next, combined, select[stage * padded_n_], 0, n_);
            AscendC::PipeBarrier<PIPE_V>();
            auto previous = values;
            values = next;
            next = previous;
        }
    }

    __aicore__ inline int32_t RoundNibble(float value) {
        int32_t q = static_cast<int32_t>(value);  // 范围约为[-7,7]，先向零截断。
        if (value >= 0.0f) {
            const float fraction = value - static_cast<float>(q);
            if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) ++q;
        } else {
            const float fraction = static_cast<float>(q) - value;
            if (fraction > 0.5f || (fraction == 0.5f && q % 2 != 0)) --q;
        }
        return q < -7 ? -7 : (q > 7 ? 7 : q);
    }

    __aicore__ inline void QuantizeRow(const AscendC::LocalTensor<float>& values,
                                      AscendC::LocalTensor<uint8_t>& packed,
                                      AscendC::LocalTensor<float>& scale) {
        float maximum = 0.0f;
        for (uint32_t i = 0; i < n_; ++i) {
            const float value = values.GetValue(i);
            const float magnitude = value < 0.0f ? -value : value;
            if (magnitude > maximum) maximum = magnitude;
        }
        // 从UB读取运行时分母，保持成功scalar Div探针的形式，不换成矢量Div/倒数乘。
        const auto divisor = divisor_buffer_.template Get<float>();
        const float seven = divisor.GetValue(0);
        const float row_scale = maximum == 0.0f ? 1.0f : maximum / seven;
        scale.SetValue(0, row_scale);
        const float stored_scale = scale.GetValue(0);
        for (uint32_t byte = 0; byte < (n_ + 1) / 2; ++byte) {
            const int32_t low = RoundNibble(values.GetValue(2 * byte) / stored_scale);
            const int32_t high = 2 * byte + 1 < n_ ? RoundNibble(values.GetValue(2 * byte + 1) / stored_scale) : 0;
            packed.SetValue(byte, static_cast<uint8_t>((low & 15) | ((high & 15) << 4)));
        }
    }

    uint32_t n_, padded_n_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> input_, output_;
    AscendC::GlobalTensor<uint8_t> packed_;
    AscendC::GlobalTensor<float> scales_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> typed_queue_, packed_queue_, scale_queue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> current_buffer_, next_buffer_, peer_buffer_, combined_buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> partner_offsets_buffer_, select_offsets_buffer_, divisor_buffer_;
};

extern "C" __global__ __aicore__ void hadamard_transform_fp16(GM_ADDR x, GM_ADDR y, uint64_t rows, uint32_t n, float scale, uint32_t method) {
    HadamardKernel<half, true, false> kernel; kernel.Run(x, y, nullptr, nullptr, rows, n, scale, method);
}
extern "C" __global__ __aicore__ void hadamard_transform_bf16(GM_ADDR x, GM_ADDR y, uint64_t rows, uint32_t n, float scale, uint32_t method) {
    HadamardKernel<bfloat16_t, true, false> kernel; kernel.Run(x, y, nullptr, nullptr, rows, n, scale, method);
}
extern "C" __global__ __aicore__ void hadamard_quantize_fp16(GM_ADDR x, GM_ADDR packed, GM_ADDR scales, uint64_t rows, uint32_t n) {
    HadamardKernel<half, false, true> kernel; kernel.Run(x, nullptr, packed, scales, rows, n, 1.0f, 0);
}
extern "C" __global__ __aicore__ void hadamard_quantize_bf16(GM_ADDR x, GM_ADDR packed, GM_ADDR scales, uint64_t rows, uint32_t n) {
    HadamardKernel<bfloat16_t, false, true> kernel; kernel.Run(x, nullptr, packed, scales, rows, n, 1.0f, 0);
}
extern "C" __global__ __aicore__ void hadamard_fused_fp16(GM_ADDR x, GM_ADDR packed, GM_ADDR scales, uint64_t rows, uint32_t n, float scale, uint32_t method) {
    HadamardKernel<half, true, true> kernel; kernel.Run(x, nullptr, packed, scales, rows, n, scale, method);
}
extern "C" __global__ __aicore__ void hadamard_fused_bf16(GM_ADDR x, GM_ADDR packed, GM_ADDR scales, uint64_t rows, uint32_t n, float scale, uint32_t method) {
    HadamardKernel<bfloat16_t, true, true> kernel; kernel.Run(x, nullptr, packed, scales, rows, n, scale, method);
}
