//
// Created by core_dump on 3/14/26.
//
#include "hadacore.hpp"

namespace hadacore
{
using namespace cute;
const int M = 16;

__device__ __constant__ uint16_t H16_fp16_bin[M * M] = {
#include "../include/h16_fp16.inc"
};
__device__ __constant__ uint16_t H16_bf16_bin[M * M] = {
#include "../include/h16_bf16.inc"
};

__device__ __constant__ half_t* H16_fp16 = (half_t*) H16_fp16_bin;
__device__ __constant__ bfloat16_t* H16_bf16 = (bfloat16_t*) H16_bf16_bin;

// 对角Hadamard矩阵
__device__ __constant__ uint16_t H2_diag_fp16_bin[M * M] = {
#include "../include/h2_diag_fp16.inc"
};
__device__ __constant__ uint16_t H2_diag_bf16_bin[M * M] = {
#include "../include/h2_diag_bf16.inc"
};
__device__ __constant__ half_t* H2_diag_fp16 = (half_t*) H2_diag_fp16_bin;
__device__ __constant__ bfloat16_t* H2_diag_bf16 = (bfloat16_t*) H2_diag_bf16_bin;

__device__ __constant__ uint16_t H4_diag_fp16_bin[M * M] = {
#include "../include/h4_diag_fp16.inc"
};
__device__ __constant__ uint16_t H4_diag_bf16_bin[M * M] = {
#include "../include/h4_diag_bf16.inc"
};
__device__ __constant__ half_t* H4_diag_fp16 = (half_t*) H4_diag_fp16_bin;
__device__ __constant__ bfloat16_t* H4_diag_bf16 = (bfloat16_t*) H4_diag_bf16_bin;

__device__ __constant__ uint16_t H8_diag_fp16_bin[M * M] = {
#include "../include/h8_diag_fp16.inc"
};
__device__ __constant__ uint16_t H8_diag_bf16_bin[M * M] = {
#include "../include/h8_diag_bf16.inc"
};
__device__ __constant__ half_t* H8_diag_fp16 = (half_t*) H8_diag_fp16_bin;
__device__ __constant__ bfloat16_t* H8_diag_bf16 = (bfloat16_t*) H8_diag_bf16_bin;

// 每个block负责计算一行
// 一次计算256，一个warp计算 CHUNKS 个256
// 一个block R_WIDTH / 256 / CHUNKS 个warp
template<typename T, int R_WIDTH, int CHUNKS>
__global__ void hadacore_less_than_4096(T* A) {

    constexpr int WARPS = R_WIDTH / 256 / CHUNKS;
    extern __shared__ __align__(16) char smemA[];
    __shared__ __align__(32) int16_t smemhada_bin1[M * M];
    __shared__ __align__(32) int16_t smemhada_bin2[M * M];

    T* smemA_total = (T*) smemA;
    T* smemhada1 = (T*)smemhada_bin1;
    T* smemhada2 = (T*)smemhada_bin2;
    T* hada1_ptr = nullptr;
    T* hada2_ptr = nullptr;
    if constexpr (std::is_same_v<cute::half_t, T>) {
        hada1_ptr = H16_fp16;
    } else {
        hada1_ptr = H16_bf16;
    }

    constexpr int log_r_width = 31 - __builtin_clz(R_WIDTH);
    if (log_r_width > 8)
    {
        if(R_WIDTH)
    }

    auto gA_total = make_tensor(
        make_gmem_ptr(A),
        make_shape(Int<WARPS * CHUNKS>{}, Int<M>{}, Int<M>{}),
        make_stride(Int<M * M>{}, Int<M>{}, Int<1>{})
    );

    auto sA_total = make_tensor(
        make_smem_ptr(smemA_total),
        make_shape(Int<WARPS * CHUNKS>{}, Int<M>{}, Int<M>{}),
        make_stride(Int<M * M>{}, Int<M>{}, Int<1>{})
    );

    auto gH1 = make_tensor(
        make_gmem_ptr(hada1_ptr),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sH1 = make_tensor(
        make_smem_ptr(smemhada1),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sH2 = make_tensor(
        make_smem_ptr(smemhada2),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sA = sA_total(threadIdx.y * CHUNKS, _, _);

    // 每个线程 load 8 个 elements
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;

    auto copyA = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<_16, _2>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto thr_copy_a = copyA.get_slice(threadIdx.x);

    auto tAgA = thr_copy_a.partition_S(gA_total(threadIdx.y * CHUNKS, _, _));
    auto tAsA = thr_copy_a.partition_D(sA);

    auto tHgH1 = thr_copy_a.partition_S(gH1);
    auto tHsH1 = thr_copy_a.partition_D(sH1);

    auto tHgH2 = thr_copy_a.partition_S(gH2);
    auto tHsH2 = thr_copy_a.partition_D(sH2);

    if (threadIdx.y == 0) {
        copy(tHgH1, tHsH1);
        if (gH2 != nullptr)
        {
            copy(tHgH2, tHsH2);
        }
    }
    if (threadIdx.x < R_WIDTH / 16)
    {
        copy(tAgA, tAsA);
    }
    __syncwarp();
    cp_async_fence();
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

    // 一个 16x8x16 atom，沿 M 方向铺 2 份 => 16x16x16
    auto mma = make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_1, _1, _1>>{},
        Layout<Shape<_1, _2, _1>>{}
    );

    for (int loop = 1; loop < CHUNKS; ++loop) {
        // 先 load 下一批 A，再计算
        auto sA_back = sA_total(threadIdx.y * CHUNKS + loop, _, _);
        auto tAgA_back = thr_copy_a.partition_S(
            gA_total(threadIdx.y * CHUNKS + loop, _, _)
        );
        auto tAsA_back = thr_copy_a.partition_D(sA_back);
        copy(tAgA_back, tAsA_back);

        cp_async_wait<0>();

        if (threadIdx.y == 0 && threadIdx.x == 0)
        {
            print_tensor(gA_total(threadIdx.y * CHUNKS + loop - 1, _, _));
            print_tensor(sA);
            print_tensor(sH1);
        }
        // 计算 H * (A * H)
        auto thr_mma = mma.get_slice(threadIdx.x);

        // 1) 右乘 H: A x H -> C
        auto tCsA = thr_mma.partition_A(sA);
        auto tCsB = thr_mma.partition_B(sH1);
        auto tCsC = thr_mma.partition_C(sA);

        auto tCrC = thr_mma.make_fragment_C(tCsC);

        clear(tCrC);
        gemm(mma, tCsA, tCsB, tCrC);
        copy(tCrC, tCsC);
        __syncwarp();
        if (threadIdx.y == 0 && threadIdx.x == 0 && loop == 1)
        {
            print_tensor(sA);
        }

        // 2) 左乘 H: H x C -> A
        auto sAt = make_tensor(sA.data(),
                make_shape(Int<M>{}, Int<M>{}),
                make_stride(Int<1>{}, Int<M>{}));
        auto tCsH = thr_mma.partition_A(sH1);
        auto tCsHC = thr_mma.partition_B(sAt);
        auto tCsC2 = thr_mma.partition_C(sA);
        auto tCrC2 = thr_mma.make_fragment_C(tCsC2);

        clear(tCrC2);
        gemm(mma, tCsH, tCsHC, tCrC2);
        copy(tCrC2, tCsC2);

        __syncwarp();
        // 完成数据 load 再进行下一批 work
        cp_async_fence();
        sA = sA_back;
    }
    cp_async_wait<0>();
    // 计算 H * (A * H)
    auto thr_mma = mma.get_slice(threadIdx.x);

    // 1) 右乘 H: A x H -> C
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sH1);
    auto tCsC = thr_mma.partition_C(sA);

    auto tCrC = thr_mma.make_fragment_C(tCsC);

    clear(tCrC);
    gemm(mma, tCsA, tCsB, tCrC);
    copy(tCrC, tCsC);
    __syncwarp();
    if (R_WIDTH < 256)
    {
        if (threadIdx.x < R_WIDTH / 16)
        {
            copy(tAsA, tAgA);
        }
        return;
    }

    // 2) 左乘 H: H x C -> A
    auto sAt = make_tensor(sA.data(),
            make_shape(Int<M>{}, Int<M>{}),
            make_stride(Int<1>{}, Int<M>{}));
    auto tCsH = thr_mma.partition_A(sH1);
    auto tCsHC = thr_mma.partition_B(sAt);
    auto tCsC2 = thr_mma.partition_C(sA);
    auto tCrC2 = thr_mma.make_fragment_C(tCsC2);

    clear(tCrC2);
    gemm(mma, tCsH, tCsHC, tCrC2);
    copy(tCrC2, tCsC2);

    auto origin_layout = make_layout(
        make_shape(Int<R_WIDTH / 256>{}, Int<256>{}),
        make_stride(Int<256>{}, Int<1>{}));
    auto new_view = make_layout(
        make_shape(Int<16>{}, Int<16>{}),
        make_stride(Int<16>{}, Int<1>{}));
    auto real_layout = composition(origin_layout, new_view);

    for (int i = 0; i < CHUNKS; ++i)
    {
        int cols = 256 / CHUNKS * WARPS;
        auto new_tensor = make_tensor(
            make_smem_ptr(smemA_total + cols), real_layout
        );

        auto tCsA = thr_mma.partition_A(new_tensor);
        auto tCsB = thr_mma.partition_B(sH2);
        auto tCsC = thr_mma.partition_C(new_tensor);

        auto tCrC = thr_mma.make_fragment_C(tCsC);

        clear(tCrC);
        gemm(mma, tCsA, tCsB, tCrC);
        copy(tCrC, tCsC);
    }

    // 需要block的全部thread同步了
    __syncthreads();
}

void test_small()
{
    constexpr int R_WIDTH = 128;  // 8 * 16
    constexpr int ROWS = R_WIDTH / M;  // 8

    // 准备输入数据 (8行16列)
    std::vector<half_t> A_h(R_WIDTH);
    for (int i = 0; i < R_WIDTH; ++i) {
        A_h[i] = half_t(i / 100.0f);
    }

    // 打印输入数据
    printf("Input A (8x16):\n");
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < M; ++c) {
            printf("%6.2f ", float(A_h[r * M + c]));
        }
        printf("\n");
    }

    // 分配 GPU 内存
    half_t *A_d, *O_d;
    cudaMalloc(&A_d, R_WIDTH * sizeof(half_t));
    cudaMalloc(&O_d, R_WIDTH * sizeof(half_t));

    // 拷贝数据到 GPU
    cudaMemcpy(A_d, A_h.data(), R_WIDTH * sizeof(half_t), cudaMemcpyHostToDevice);

    // 调用 kernel
    hada_core_less_256<half_t, R_WIDTH><<<1, 32>>>(A_d, O_d);

    // 等待完成
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    std::vector<half_t> result(R_WIDTH);
    cudaMemcpy(result.data(), A_d, R_WIDTH * sizeof(half_t), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("\nOutput A after H * A * H (8x16):\n");
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < M; ++c) {
            printf("%6.2f ", float(result[r * M + c]));
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(A_d);
    cudaFree(O_d);
}

void test_large()
{
    constexpr int R_WIDTH = 1024;  // 总行宽
    constexpr int CHUNKS = 2;     // 每个 warp 处理的 chunk 数
    constexpr int WARPS = R_WIDTH / 256 / CHUNKS;  // = 2

    // 准备输入数据 (512 = 32行 x 16列)
    std::vector<half_t> A_h(R_WIDTH);
    for (int i = 0; i < R_WIDTH; ++i)
    {
        A_h[i] = half_t(i / 100.0f);  // 0,1,2,...,15,0,1,2,...
    }
    // 分配 GPU 内存
    half_t *A_d;
    cudaMalloc(&A_d, R_WIDTH * sizeof(half_t));

    // 拷贝数据到 GPU
    cudaMemcpy(A_d, A_h.data(), R_WIDTH * sizeof(half_t), cudaMemcpyHostToDevice);

    // 计算 dynamic shared memory 大小
    // 每个 chunk 是 16x16，每个 warp 处理 CHUNKS 个
    size_t smem_size = std::max(WARPS * CHUNKS * M * M * sizeof(half_t), 16 * sizeof(half_t));

    printf("\nLaunching kernel: R_WIDTH=%d, CHUNKS=%d, WARPS=%d\n", R_WIDTH, CHUNKS, WARPS);
    printf("Block dim: (%d, %d, 1), Dynamic smem: %zu bytes\n\n", 32, WARPS, smem_size);

    // 调用 kernel
    dim3 block(32, WARPS);
    hadacore_less_than_4096<half_t, R_WIDTH, CHUNKS><<<1, block, smem_size>>>(A_d);

    // 等待完成
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    std::vector<half_t> result(R_WIDTH);
    cudaMemcpy(result.data(), A_d, R_WIDTH * sizeof(half_t), cudaMemcpyDeviceToHost);

    // 打印结果 (前32行)
    printf("Output A after H * A * H (first 32x16):\n");
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < M; ++c) {
            printf("%6.1f ", float(result[r * M + c]));
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(A_d);
}
}

