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

// 处理 64 < R_WIDTH < 256
template<typename T, int R_WIDTH>
__global__ void hada_core_less_256(T* A, T* O_scope) {

    constexpr int ROWS = R_WIDTH / M;

    __shared__ __align__(32) int16_t smemA_bin[M * M];
    __shared__ __align__(32) int16_t smemhada_bin[M * M];
    __shared__ __align__(32) int16_t smemC_bin[M * M];   // 暂存 A * H

    T* smemA    = (T*) smemA_bin;
    T* smemC    = (T*) smemC_bin;
    T* smemhada = (T*) smemhada_bin;

    T* hada_ptr = nullptr;

    if constexpr (std::is_same_v<cute::half_t, T>) {
        hada_ptr = H16_fp16;
    } else {
        hada_ptr = H16_bf16;
    }

    auto gA = make_tensor(
        make_gmem_ptr(A),
        make_shape(Int<ROWS>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sA = make_tensor(
        make_smem_ptr(smemA),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto thr_sA = local_partition(
        sA,
        Layout<Shape<_32>>{},
        threadIdx.x
    );

    clear(thr_sA);   // 清空
    __syncthreads();

    auto gH = make_tensor(
        make_gmem_ptr(hada_ptr),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sH = make_tensor(
        make_smem_ptr(smemhada),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sC = make_tensor(
        make_smem_ptr(smemC),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, T>;

    auto copyA = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<ROWS>, _2>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto copyH = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<_16, _2>>{},
        Layout<Shape<_1, _8>>{}
    );

    auto sA_sub = local_tile(
        sA,
        make_shape(Int<ROWS>{}, Int<M>{}),
        make_coord(0, 0)
    );

    auto thr_copy_a = copyA.get_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = thr_copy_a.partition_D(sA_sub);

    auto thr_copy_h = copyH.get_slice(threadIdx.x);
    auto tHgH = thr_copy_h.partition_S(gH);
    auto tHsH = thr_copy_h.partition_D(sH);

    copy(tAgA, tAsA);
    copy(tHgH, tHsH);

    if (threadIdx.x == 0) {
        print_tensor(sA);
        print_tensor(sH);
    }

    __syncthreads();

    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

    // 一个 16x8x16 atom，沿 M 方向铺 2 份 => 16x16x16
    auto mma = make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_1, _1, _1>>{},
        Layout<Shape<_1, _2, _1>>{}
    );

    auto thr_mma = mma.get_slice(threadIdx.x);

    // -------------------------------
    // 1) 右乘 H:  A x H -> C
    // -------------------------------

    auto tCsA = thr_mma.partition_A(sA);   // A
    auto tCsB = thr_mma.partition_B(sH);   // H
    auto tCsC = thr_mma.partition_C(sC);   // C

    auto tCrC = thr_mma.make_fragment_C(tCsC);
    clear(tCrC);

    gemm(mma, tCsA, tCsB, tCrC);

    copy(tCrC, tCsC);

    __syncthreads();

    if (threadIdx.x == 0) {
        print_tensor(sC);
    }

    auto sCt = make_tensor(
        make_smem_ptr(smemC),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<1>{}, Int<M>{})
    );   // sC 的转置 view

    auto t2sA = thr_mma.partition_A(sH);
    auto t2sB = thr_mma.partition_B(sCt);
    auto t2sC = thr_mma.partition_C(sA);

    auto t2rC = thr_mma.make_fragment_C(t2sC);
    clear(t2rC);

    gemm(mma, t2sA, t2sB, t2rC);

    copy(t2rC, t2sC);
    __syncthreads();

    if (threadIdx.x == 0) {
        print_tensor(sA);   // row-major 查看结果
    }
    copy(tAsA, tAgA);

    __syncthreads();
}

// 每个block负责计算一行
// 一次计算256，一个warp计算 CHUNKS 个256
// 一个block R_WIDTH / 256 / CHUNKS 个warp
template<typename T, int R_WIDTH, int CHUNKS>
__global__ void hadacore_large(const T* A) {

    constexpr int WARPS = R_WIDTH / 256 / CHUNKS;
    extern __shared__ __align__(16) char smemA[];
    __shared__ __align__(32) int16_t smemhada_bin[M * M];

    T* smemA_total = (T*) smemA;
    T* smemhada = (T*)smemhada_bin;
    T* hada_ptr = nullptr;
    if constexpr (std::is_same_v<cute::half_t, T>) {
        hada_ptr = H16_fp16;
    } else {
        hada_ptr = H16_bf16;
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

    auto gH = make_tensor(
        make_gmem_ptr(hada_ptr),
        make_shape(Int<M>{}, Int<M>{}),
        make_stride(Int<M>{}, Int<1>{})
    );

    auto sH = make_tensor(
        make_smem_ptr(smemhada),
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

    auto tHgH = thr_copy_a.partition_S(gH);
    auto tHsH = thr_copy_a.partition_D(sH);

    if (threadIdx.y == 0) {
        copy(tHgH, tHsH);
    }
    copy(tAgA, tAsA);
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
            print_tensor(sH);
        }
        // 计算 H * (A * H)
        auto thr_mma = mma.get_slice(threadIdx.x);

        // 1) 右乘 H: A x H -> C
        auto tCsA = thr_mma.partition_A(sA);
        auto tCsB = thr_mma.partition_B(sH);
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
        auto tCsH = thr_mma.partition_A(sH);
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
    auto tCsB = thr_mma.partition_B(sH);
    auto tCsC = thr_mma.partition_C(sA);

    auto tCrC = thr_mma.make_fragment_C(tCsC);

    clear(tCrC);
    gemm(mma, tCsA, tCsB, tCrC);
    copy(tCrC, tCsC);
    __syncwarp();

    // 2) 左乘 H: H x C -> A
    auto sAt = make_tensor(sA.data(),
            make_shape(Int<M>{}, Int<M>{}),
            make_stride(Int<1>{}, Int<M>{}));
    auto tCsH = thr_mma.partition_A(sH);
    auto tCsHC = thr_mma.partition_B(sAt);
    auto tCsC2 = thr_mma.partition_C(sA);
    auto tCrC2 = thr_mma.make_fragment_C(tCsC2);

    clear(tCrC2);
    gemm(mma, tCsH, tCsHC, tCrC2);
    copy(tCrC2, tCsC2);

    // 需要block的全部thread同步了
    __syncthreads();

    new

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
    size_t smem_size = WARPS * CHUNKS * M * M * sizeof(half_t);

    printf("\nLaunching kernel: R_WIDTH=%d, CHUNKS=%d, WARPS=%d\n", R_WIDTH, CHUNKS, WARPS);
    printf("Block dim: (%d, %d, 1), Dynamic smem: %zu bytes\n\n", 32, WARPS, smem_size);

    // 调用 kernel
    dim3 block(32, WARPS);
    hadacore_large<half_t, R_WIDTH, CHUNKS><<<1, block, smem_size>>>(A_d);

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

