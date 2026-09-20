// topk_merge.cuh - 精确检索与 IVF 查询共用的 Top-K 归并
//
// 前提：每个线程在本地维护一个“已按优到劣排序”的 top-K（长度 K，含哨兵项）。
// 归并利用“各列表内部已有序”这一性质，只需 K 轮“取全局最优”，
// 总代价 O(K · log(线程数))，避免对全部候选做排序。
//
// “更优”判定采用全序：分数更优者胜；分数相等时 id 小者胜。
// 这样 GPU 结果与 CPU 参考（稳定插入）完全一致。
#pragma once
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>

namespace vs {

__device__ __forceinline__ bool is_better_d(float sa, int32_t ia, float sb, int32_t ib,
                                            bool smaller) {
    if (smaller) {
        if (sa < sb)
            return true;
        if (sa > sb)
            return false;
    } else {
        if (sa > sb)
            return true;
        if (sa < sb)
            return false;
    }
    return ia < ib;
}

// 两级归并（warp 内 32 路 → 跨 warp），把整个 block 的 top-K 写到
// out_d/out_i 的 [out_base, out_base + K)。
// 共享内存约定：s_wd / s_wi 各需 NWARP*K 个元素。
template <int K, int THREADS>
__device__ __forceinline__ void block_topk_merge(const float* bd, const int32_t* bi, bool smallerb,
                                                 float* s_wd, int32_t* s_wi, float* out_d,
                                                 int32_t* out_i, int64_t out_base) {
    constexpr int NWARP = THREADS / 32;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float WORST = smallerb ? FLT_MAX : -FLT_MAX;

    // ---- warp 级：32 个 lane 的有序表 → 本 warp 的 top-K ----
    {
        int head = 0;
        for (int r = 0; r < K; r++) {
            float mine = (head < K) ? bd[head] : WORST;
            int32_t mid = (head < K) ? bi[head] : INT_MAX;
            float v = mine;
            int32_t vid = mid;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, v, off, 32);
                int32_t oid = __shfl_xor_sync(0xffffffffu, vid, off, 32);
                if (is_better_d(ov, oid, v, vid, smallerb)) {
                    v = ov;
                    vid = oid;
                }
            }
            if (lane == (r & 31)) {
                s_wd[warp * K + r] = v;
                s_wi[warp * K + r] = vid;
            }
            // 持有该最优项的 lane 前进一格
            if (head < K && bi[head] == vid)
                head++;
        }
    }
    __syncthreads();

    // ---- 跨 warp：NWARP 路 → 最终 top-K ----
    // 注意：只有 warp0 的前 NWARP 个 lane 参与，因此 __shfl_xor_sync 的 mask
    // 必须精确等于这些 lane，否则属于未定义行为（曾因此导致结果时对时错）。
    if (warp == 0 && lane < NWARP) {
        constexpr unsigned kMask = (1u << NWARP) - 1u;
        int head = 0;
        for (int r = 0; r < K; r++) {
            float mine = (head < K) ? s_wd[lane * K + head] : WORST;
            int32_t mid = (head < K) ? s_wi[lane * K + head] : INT_MAX;
            float v = mine;
            int32_t vid = mid;
#pragma unroll
            for (int off = NWARP / 2; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(kMask, v, off, NWARP);
                int32_t oid = __shfl_xor_sync(kMask, vid, off, NWARP);
                if (is_better_d(ov, oid, v, vid, smallerb)) {
                    v = ov;
                    vid = oid;
                }
            }
            if (lane == (r % NWARP)) {
                out_d[out_base + r] = v;
                out_i[out_base + r] = (vid == INT_MAX) ? -1 : vid;
            }
            if (head < K && s_wi[lane * K + head] == vid)
                head++;
        }
    }
}

} // namespace vs
