//
// Created by flashzxi on 2/24/26.
//
#include <cuda_fp16.h>
#include <string>
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/numeric_types.h"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#define ENABLE_DOUBLE_BUFFER

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

struct QuantState {
    // header
    int num_rows;
    int num_cols;
    int block_size;
    int group_size;

    // data
    uint8_t* packed_weights;   // 每字节存两个 4-bit 索引
    uint8_t* absmax_q;
    __half* absmax2;
    __half code2[256]; // 二级码表
    float offset;

    // runtime param
    std::string compute_type;
    std::string target_gpu;

    int num_elements;
    int num_blocks;
    int num_groups;

    int packed_weights_len_in_bytes;
    int absmax_q_len_in_bytes;
    int absmax2_len_in_bytes;

    // 输出位置
    uint8_t *output;

    void calculate_params() {
        num_elements = num_rows * num_cols;
        group_size = block_size;
        num_blocks = (num_elements + block_size - 1) / block_size;
        num_groups = (num_blocks + group_size - 1) / group_size;

        packed_weights_len_in_bytes = (num_elements + 1) / 2;
        absmax_q_len_in_bytes = num_blocks;
        absmax2_len_in_bytes = 2 * num_groups;
    }
};

// code2 为 256 * f16
// 每个线程load 2 个，需要128个线程， 故设置一个block 128个线程，每个线程处理N个计算
// 总计处理128 * N个数据, N 是2的幂 且不小于8
// 结尾不够需要padding
template<typename FP_T, int N>
__global__ void dequant_nf4_scale_f16xN_kernel(uint8_t* scale_q, FP_T* code2, FP_T* absmax2, int num_blocks, int group_size, FP_T* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x;

    // load code2
    __shared__ float shm_code2[128];
    shm_code2[lane_id] = code2[lane_id];

    // 一次处理8个数据
    constexpr int loop_times = N / 8;
    int g_scale_q_offset_base = blockIdx.x * 128 * N;

    // 使用double buffer需要使用shared_memory
    // 不使用double buffer 可以直接再寄存器上暂存数据
#ifdef ENABLE_DOUBLE_BUFFER
    auto block = cooperative_groups::this_thread_block();

    constexpr int STAGES = 2;  // 双缓冲
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> ps;
    auto pipe = cuda::make_pipeline(block, &ps);

    // 一次读取8个数据, 双buffer
    __shared__ uint8_t fragment[2][128][8];
    // 取第0块
    int scale_offset = g_scale_q_offset_base + lane_id * 8;
    if (scale_offset + 8 <= num_blocks) {
        pipe.producer_acquire();
        cuda::memcpy_async(block, fragment[0][lane_id], scale_q + scale_offset, cuda::aligned_size_t<16>(16), pipe);
        pipe.producer_commit();
    } else if (scale_offset < num_blocks) {
      // 尾块处理：退化为标量copy
      for (int i = 0; i < 8 && scale_offset + i < num_blocks; ++i) {
          fragment[0][lane_id][i] = scale_q[scale_offset + i];
      }
    }

#pragma unroll
    for (int i = 0; i < loop_times; ++i) {

    }
#else
    uint8_t fragment[8];
    FP_T cache_res[8];
#pragma unroll
    for (int i = 0; i < loop_times; ++i) {
        int scale_offset = g_scale_q_offset_base + i * 128 * 8 + lane_id * 8;
        FP_T scale2 = absmax2[i * 128 * 8 + lane_id * 8 / group_size];
        if (scale_offset + 7 < num_blocks) {
            LDST64BITS(fragment[0]) = LDST64BITS(*(scale_q + scale_offset));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cache_res[j] = ((FP_T*) shm_code2)[fragment[i]] * scale2;
            }
            LDST128BITS(output[scale_offset]) = LDST128BITS(cache_res[0]);
        } else if (scale_offset < num_blocks) {
            // 不够一组，退化为每个元素load
            int remains = num_blocks - scale_offset;
            for (int j = 0; j < remains; ++j) {
                fragment[j] = (scale_q + scale_offset)[j];
                cache_res[j] = ((FP_T*) shm_code2)[fragment[i]] * scale2;
                output[scale_offset + j] = cache_res[j];
            }
        }
    }
#endif
}

void nf4_dequant(const QuantState& quant_state) {
    // 解码scale
    cutlass::HostTensor<uint8_t, cutlass::layout::PitchLinear> scale_q(cutlass::layout::PitchLinearCoord(quant_state.num_blocks, 1));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::PitchLinear> code2(cutlass::layout::PitchLinearCoord(256, 1));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::PitchLinear> absmax2(cutlass::layout::PitchLinearCoord(quant_state.num_groups, 1));

    memcpy(scale_q.host_data(), quant_state.absmax_q, quant_state.absmax_q_len_in_bytes);
    memcpy(code2.host_data(), quant_state.code2, 256 * 2);
    memcpy(absmax2.host_data(), quant_state.absmax2, quant_state.absmax2_len_in_bytes);

    scale_q.sync_device();
    code2.sync_device();
    absmax2.sync_device();

    constexpr int dequant_scale_per_thread_calc = 8;
    dim3 dequant_scale_block_dim(128);
    dim3 dequant_scale_grid_dim((quant_state.block_size + 128 * dequant_scale_per_thread_calc - 1) / 128 * dequant_scale_per_thread_calc);
    // 解码权重
}