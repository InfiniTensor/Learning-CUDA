#include <cuda_runtime.h>

#include <cstddef>

#define WARPSIZE 32

// 感觉用了才觉得这个是必须的，不然要对外写很多模板的细节要求或者写很多的宏定义
// 且这些要求一般都是内含计算规律和要求的，以便应用到kernel中的计算，内存申请中
template <int HeadDim, int Br, int Bc, int NumWarps, bool Causal>
struct Fa2Traits
{
    static constexpr int kWarpSize = WARPSIZE;
    static constexpr int kThreads = NumWarps * kWarpSize;

    static_assert(HeadDim > 0 && HeadDim % kWarpSize == 0);
    static_assert(Br > 0 && Br % NumWarps == 0);
    static_assert(Bc > 0 && Bc % kWarpSize == 0);

    static constexpr int kDPerLane = HeadDim / kWarpSize;
    static constexpr int kRowsPerWarp = Br / NumWarps;
    static constexpr int kKeysPerLane = Bc / kWarpSize;
    static constexpr int kKvTileElements = Bc * HeadDim;
    static constexpr std::size_t kSmemBytes =
        2 * kKvTileElements * sizeof(float);
};

struct Fa2Params
{
    const float *q;
    const float *k;
    const float *v;
    float *o;

    int batch_size;
    int q_heads;
    int kv_heads;
    int q_lens;
    int kv_lens;

    float qk_scale; // 一般是1/sqrt(head_Dim)
};

// 用xor有更好的效果
__device__ __forceinline__ float
warpReduceSum(float v)
{
    unsigned Mask = 0xffffffffu;
    v += __shfl_xor_sync(Mask, v, 16);
    v += __shfl_xor_sync(Mask, v, 8);
    v += __shfl_xor_sync(Mask, v, 4);
    v += __shfl_xor_sync(Mask, v, 2);
    v += __shfl_xor_sync(Mask, v, 1);

    return v;
}

__device__ __forceinline__ float warpReduceMax(float v)
{
    unsigned Mask = 0xffffffffu;

    v = fmaxf(v, __shfl_xor_sync(Mask, v, 16));
    v = fmaxf(v, __shfl_xor_sync(Mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(Mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(Mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(Mask, v, 1));

    return v;
}

// 模板是为了让编译器在编译期间就确定好代码的基本结构
// 比如可以让smem可以使用模板来去创建内存大小
// 比如可以根据模板的情况 来去删去/保留某些代码流
//***一个Warp处理一个Row的Dim个数据(一般一个Warp负责多个Rows，但是是一个Row处理完在处理下一个)
// 每个Lane负责处理和保存的元素 = HeadDim / WARPSIZE

template <int Br, int Bc, int HeadDim, int NumWarps, bool Causal>
__global__ void FlashAttentionKernel(Fa2Params params)
{
    using Traits = Fa2Traits<HeadDim, Br, Bc, NumWarps, Causal>;
    // params.qk_scale = 1 / sqrt(HeadDim);
    int tid = threadIdx.x;

    // Block的并行维度
    int bid = blockIdx.x;
    int q_head = blockIdx.y;
    int batch = blockIdx.z;

    int query_heads = params.q_heads;
    int kv_heads = params.kv_heads;

    int group_size = query_heads / kv_heads; // GQA/MHA
    int kv_head = q_head / group_size;       // 让多个q_head共享使用第i个kv_head来实现GQA 体现在blockIdx.y

    int warp_id = tid / WARPSIZE;
    int warp_lane = tid % WARPSIZE; // 当前线程属于warp内的哪一个线程

    // 外层需要遍历KV_tile 同时需要知道当前Block需要处理的Q_tile是哪一个内存偏移
    // KV_tile的地址是会变化读取的，而Q_tile在读取后对当前Block是静态的

    // Block内的warp处理好几个rows，需要为每个warp自己开始处理的row进行内部定位
    // 后续在KV tile循环的过程中，他们的rows会一次一行的处理完，这时候我们开始记录内部的Q_row的offset
    int Q_row_base = bid * Br;
    int Q_row_offset;

    __shared__ float smem[2 * Traits::kKvTileElements];
    float *k_tile = smem;
    float *v_tile = smem + Traits::kKvTileElements;

    float m_reg[Traits::kRowsPerWarp]; // 每行的最大值(Score中的最大值)
    float l_reg[Traits::kRowsPerWarp]; // softmax每行的指数和分母
    float acc_reg[Traits::kRowsPerWarp][Traits::kDPerLane];
    float q_tile[Traits::kRowsPerWarp][Traits::kDPerLane];

    // 1.进行Q tile的加载等初始化过程
    for (int rowIdx = 0; rowIdx < Traits::kRowsPerWarp; rowIdx++)
    {
        // 计算自己warp负责的rows的起始偏移行(Block tile内部的row坐标计算了)
        // 正确地址写法（沿用当前变量名）：
        // int row_in_warp = warp_id * Traits::kRowsPerWarp + rowIdx;
        // Q_row_offset = Q_row_base + row_in_warp;
        // std::size_t Q_base =
        //     ((static_cast<std::size_t>(batch) * query_heads + q_head) *
        //          params.q_lens +
        //      Q_row_offset) *
        //     HeadDim;
        int row_in_warp = warp_id * Traits::kRowsPerWarp + rowIdx;
        // 主要这个是在计算Q[,,Len,] 第三维度对应的token的
        Q_row_offset = row_in_warp + Q_row_base; // Q[B,Hq,Len,D] Stride[Hq*Len*D,Len*D,D,1]
        const std::size_t Q_base =
            ((static_cast<std::size_t>(batch) * query_heads + q_head) *
                 params.q_lens +
             Q_row_offset) *
            HeadDim;
        // 基于Q_base的移动就是在遍历某个特定Head的Dim

        for (int colIdx = 0; colIdx < Traits::kDPerLane; colIdx++)
        {
            int d = warp_lane + WARPSIZE * colIdx; // 对于线程角度下的k这个实际索引 lane0:0--32--64 lane1:1--33--65
            q_tile[rowIdx][colIdx] = params.q[Q_base + d];
            acc_reg[rowIdx][colIdx] = 0.0f;
        }
        m_reg[rowIdx] = -INFINITY;
        l_reg[rowIdx] = 0.0f;
    }

    // 目前是要求需要整除， 没太做一些冗余一小部分的padding处理
    int kv_tile_nums = params.kv_lens / Bc;
    // 2.开始主要的KV tile的循环

    for (int kv_tile_idx = 0; kv_tile_idx < kv_tile_nums; kv_tile_idx++)
    {

        int kv_tile_base = kv_tile_idx * Bc;
        // Block内的thread需要协作从Bc x d的KV tile中联合读取数据
        // 就采用Block内线程联合读取到smem的方式来去处理
        // 正确地址写法（K/V 的布局相同，共用同一个起始偏移）：
        // std::size_t KV_base =
        //     ((static_cast<std::size_t>(batch) * kv_heads + kv_head) *
        //          params.kv_lens +
        //      kv_tile_base) *
        //     HeadDim;
        const std::size_t KV_base =
            ((static_cast<std::size_t>(batch) * kv_heads + kv_head) *
                 params.kv_lens +
             kv_tile_base) *
            HeadDim;
        for (int i = tid; i < Traits::kKvTileElements; i += Traits::kThreads)
        {
            k_tile[i] = params.k[KV_base + i];
            v_tile[i] = params.v[KV_base + i];
            // 当前KV tile循环下所需要的数据都已加载到smem中
        }
        __syncthreads();
        for (int rowIdx = 0; rowIdx < Traits::kRowsPerWarp; rowIdx++) // 这里的是为了计算BrxBc大小的score
        {
            float score_and_p[Traits::kKeysPerLane]; // 每行的数据都进行一次记录，他主要是作为中转后续存到acc中
            // 同一 warp 内各 lane 分散持有不同 key 的 score/p；它们不是相同副本。
            for (int i = 0; i < Traits::kKeysPerLane; i++)
                score_and_p[i] = -INFINITY; // 这里不是0
            // 该

            for (int k = 0; k < Bc; k++) // 内部的一次for循环求一个dot的score，这里的循环是为了计算一行的dot的score
            {
                float partial_dot = 0.0f;
                for (int colIdx = 0; colIdx < Traits::kDPerLane; colIdx++)
                // k---这个代表着在处理一行中的第几个score，这个score的位置需要由对此负责的warp中的一个线程来去认领和处理
                {

                    int d = warp_lane + colIdx * WARPSIZE;
                    partial_dot += q_tile[rowIdx][colIdx] * k_tile[k * HeadDim + d]; // k在K_tile中是row坐标
                }

                float qk_dot = warpReduceSum(partial_dot);
                bool is_Mask = true; // true时对应的是q位置合法
                if constexpr (Causal)
                {
                    //
                    int row_In_warp = warp_id * Traits::kRowsPerWarp + rowIdx;
                    int q_postion = Q_row_base + row_In_warp; // 目前是这个pos的设置
                    int k_postion = k + kv_tile_base;
                    is_Mask = k_postion <= q_postion; // causal：只能访问当前位置及以前的 key
                }
                int owner_slot = k / WARPSIZE; // 这里slot用来表示 负责处理这个lane的线程 将其保存到他的寄存器的第一个槽位中
                int owner_lane = k % WARPSIZE; // 这里并行的是多个warp来去协调处理当前与KV_tile的结果
                if (owner_lane == warp_lane)
                {
                    score_and_p[owner_slot] = is_Mask ? params.qk_scale * qk_dot : -INFINITY; // 此时还没有经过softmax的指数处理
                    // -INFINITY是指数后正好为0
                }
            }

            float local_tile_max = -INFINITY;

            for (int colIdx = 0; colIdx < Traits::kKeysPerLane; colIdx++)
            {
                local_tile_max = fmaxf(local_tile_max, score_and_p[colIdx]);
            }

            // 这里有点关于需不需要专门线程来去存储/处理最大值的一个小问题
            float tile_max = warpReduceMax(local_tile_max);
            float m_old = m_reg[rowIdx];
            float m_new = fmaxf(m_old, tile_max);
            float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new); // 这里的条件是在预防什么情况
            m_reg[rowIdx] = m_new;

            float local_tile_sum = 0.0f;

            for (int colIdx = 0; colIdx < Traits::kKeysPerLane; colIdx++)
            {
                float score = score_and_p[colIdx];
                float p = (score == -INFINITY) ? 0.0f : __expf(score - m_new); // 这部分的公式稍微确认一下
                score_and_p[colIdx] = p;
                local_tile_sum += p;
            }
            float tile_sum = warpReduceSum(local_tile_sum);
            l_reg[rowIdx] = alpha * l_reg[rowIdx] + tile_sum;

            for (int colIdx = 0; colIdx < Traits::kDPerLane; colIdx++)
            {
                acc_reg[rowIdx][colIdx] = alpha * acc_reg[rowIdx][colIdx];
            }

            for (int colIdx = 0; colIdx < Traits::kKeysPerLane; colIdx++)
            {
                for (int owner_lane = 0; owner_lane < WARPSIZE; owner_lane++)
                {
                    int key = colIdx * WARPSIZE + owner_lane;
                    float p = __shfl_sync(0xffffffffu, score_and_p[colIdx], owner_lane);

                    for (int d_slot = 0; d_slot < Traits::kDPerLane; d_slot++)
                    {
                        int d = warp_lane + d_slot * WARPSIZE;
                        acc_reg[rowIdx][d_slot] += (p * v_tile[key * HeadDim + d]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // 此时KV tile迭代完毕 所有的acc_reg中存有的最终的o的结果，需要各自将其写入对应的位置中
    // 一行一行的写入
    for (int row_Idx = 0; row_Idx < Traits::kRowsPerWarp; row_Idx++)
    { // O[B,Hq,Len,D] Stride[Hq*Len*D,Len*D,D,1]

        // 正确地址写法（当前 row_Idx 对应的全局 Q/O 行）：
        // int row_in_warp =
        //     warp_id * Traits::kRowsPerWarp + row_Idx;
        // Q_row_offset = Q_row_base + row_in_warp;
        // std::size_t O_base =
        //     ((static_cast<std::size_t>(batch) * query_heads + q_head) *
        //          params.q_lens +
        //      Q_row_offset) *
        //     HeadDim;
        int row_in_warp = warp_id * Traits::kRowsPerWarp + row_Idx;
        Q_row_offset = Q_row_base + row_in_warp;
        const std::size_t O_base =
            ((static_cast<std::size_t>(batch) * query_heads + q_head) *
                 params.q_lens +
             Q_row_offset) *
            HeadDim;

        float inv_l = l_reg[row_Idx] > 0.0f ? 1.0f / l_reg[row_Idx] : 0.0f; // 分母指数和 后面直接作为系数调整acc即可
        // l_reg 在所有 lane 中是相同副本；每个 lane 用它归一化自己负责的 d。

        for (int col_Idx = 0; col_Idx < Traits::kDPerLane; col_Idx++)
        {
            int d = warp_lane + col_Idx * WARPSIZE;
            params.o[O_base + d] = acc_reg[row_Idx][col_Idx] * inv_l;
        }
    }
}

// Q/O 使用 [B,Hq,Lq,D] 坐标；K/V 使用 [B,Hkv,Lkv,D] 坐标。
