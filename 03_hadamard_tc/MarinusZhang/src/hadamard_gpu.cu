// GPU entry points: the M0 toolchain probe, the M2 butterfly FWHT baseline, the M3
// Tensor Core GEMM, and the M4 FP8 E4M3 quantizers (standalone and fused).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "hadamard/cuda_utils.h"
#include "hadamard/hadamard.h"

namespace hadamard {
namespace {

__global__ void hello_kernel(int* out, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = value;
    }
}

// Half/bfloat16 helpers so the probe, the FWHT kernel and the epilogue can be
// written once for both element types.
template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    static __host__ __device__ __half from_float(float v) { return __float2half_rn(v); }
    static __host__ __device__ float to_float(__half v) { return __half2float(v); }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static __host__ __device__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
    static __host__ __device__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// C = A * B for one 16x16x16 WMMA tile. Only used to prove the Tensor Core path
// works for a given element type; the real kernel comes in M3.
template <typename T>
__global__ void wmma_identity_probe_kernel(const T* a, const T* b, float* c) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> fa;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major> fb;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> fc;
    nvcuda::wmma::fill_fragment(fc, 0.0f);
    nvcuda::wmma::load_matrix_sync(fa, a, 16);
    nvcuda::wmma::load_matrix_sync(fb, b, 16);
    nvcuda::wmma::mma_sync(fc, fa, fb, fc);
    nvcuda::wmma::store_matrix_sync(c, fc, 16, nvcuda::wmma::mem_row_major);
}

// A is filled with 0..255 (exact in fp16 and bf16) and B with the identity, so the
// expected result is exactly A.
template <typename T>
bool run_identity_probe(bool* correct, cudaStream_t stream) {
    constexpr int kDim = 16;
    constexpr int kElems = kDim * kDim;
    std::vector<T> h_a(kElems);
    std::vector<T> h_b(kElems);
    std::vector<float> h_expected(kElems);
    std::vector<float> h_got(kElems, 0.0f);
    for (int i = 0; i < kDim; ++i) {
        for (int j = 0; j < kDim; ++j) {
            const int idx = i * kDim + j;
            h_a[idx] = TypeTraits<T>::from_float(static_cast<float>(idx));
            h_b[idx] = TypeTraits<T>::from_float(i == j ? 1.0f : 0.0f);
            h_expected[idx] = static_cast<float>(idx);
        }
    }

    T* d_a = nullptr;
    T* d_b = nullptr;
    float* d_c = nullptr;
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(T) * kElems));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(T) * kElems));
    HW_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * kElems));
    HW_CUDA_CHECK(
        cudaMemcpyAsync(d_a, h_a.data(), sizeof(T) * kElems, cudaMemcpyHostToDevice, stream));
    HW_CUDA_CHECK(
        cudaMemcpyAsync(d_b, h_b.data(), sizeof(T) * kElems, cudaMemcpyHostToDevice, stream));
    wmma_identity_probe_kernel<T><<<1, 32, 0, stream>>>(d_a, d_b, d_c);
    HW_CUDA_CHECK(cudaGetLastError());
    HW_CUDA_CHECK(
        cudaMemcpyAsync(h_got.data(), d_c, sizeof(float) * kElems, cudaMemcpyDeviceToHost, stream));
    HW_CUDA_CHECK(cudaStreamSynchronize(stream));
    HW_CUDA_CHECK(cudaFree(d_a));
    HW_CUDA_CHECK(cudaFree(d_b));
    HW_CUDA_CHECK(cudaFree(d_c));

    bool ok = true;
    for (int i = 0; i < kElems; ++i) {
        if (h_got[i] != h_expected[i]) {
            std::fprintf(stderr, "wmma probe mismatch at %d: got %f expected %f\n", i, h_got[i],
                         h_expected[i]);
            ok = false;
            break;
        }
    }
    *correct = ok;
    return true;
}

bool required_arch_ok(DType dtype) {
    int major = 0;
    int minor = 0;
    HW_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    HW_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    const bool supports_fp16_wmma = (major * 10 + minor) >= 70;
    if (!supports_fp16_wmma) {
        return false;
    }
    // bf16 WMMA landed with Ampere (sm_80).
    return dtype == DType::kFp16 || (major * 10 + minor) >= 80;
}

// ---------------------------------------------------------------------------
// M2: butterfly FWHT, all stages in registers
//
// Convention: y = H_d * x / sqrt(d) with H_d the Sylvester matrix. This is what
// the CPU reference implements, and it is the convention of the reference library
// (Dao-AILab/fast-hadamard-transform calls its own kernel with scale = 1/sqrt(dim)
// and documents it as F.linear(x, scipy.linalg.hadamard(dim)) * scale).
//
// A row of length d is owned by T threads of the same warp; thread `lane` holds
// the contiguous run [lane * E, lane * E + E) with E = d / T. The butterfly stage
// with stride `len` pairs index i with i ^ len:
//   * len <  E: both indices are in this thread's own registers -> plain adds;
//   * len >= E: the partner is lane ^ (len / E) at the same element index, so one
//     __shfl_xor_sync per element is enough, with the sign folded into the sum.
// Every stage therefore runs out of registers: no shared memory, no
// __syncthreads(), and a row crosses the memory bus exactly twice. Accumulation is
// FP32 and 1/sqrt(d) is folded into the epilogue, so the result is rounded to
// fp16/bf16 only once.
// ---------------------------------------------------------------------------

constexpr int kMaxThreadsPerBlock = 256;

// Blocks we would like to keep in flight: for the shapes of this assignment many
// small blocks beat a few large ones.
constexpr int kTargetBlocksPerSm = 8;

// Elements per thread is a power of two and E = 8 is a 128-bit vector for both
// supported element types, so that is the default. Rows wider than 8 * 32 grow E
// beyond it instead of T, because a shuffle group cannot span more than one warp.
constexpr int preferred_log_t(int log_d) {
    return log_d >= 8 ? 5 : (log_d >= 3 ? log_d - 3 : 0);
}

// Vectorised load/store of the E elements a thread owns, using the widest vector
// that divides the run (16 B, then 8 B, 4 B, 2 B for the degenerate shapes).
template <typename T, int E>
struct RunIO {
    static constexpr int kRunBytes = E * static_cast<int>(sizeof(T));

    static __device__ __forceinline__ void load(const T* __restrict__ src, float (&v)[E]) {
        if constexpr (kRunBytes % 16 == 0) {
            load_vec<uint4>(src, v);
        } else if constexpr (kRunBytes % 8 == 0) {
            load_vec<uint64_t>(src, v);
        } else if constexpr (kRunBytes % 4 == 0) {
            load_vec<uint32_t>(src, v);
        } else {
            load_vec<uint16_t>(src, v);
        }
    }

    static __device__ __forceinline__ void store(T* __restrict__ dst, const float (&v)[E],
                                                 float scale) {
        T rounded[E];
#pragma unroll
        for (int i = 0; i < E; ++i) {
            rounded[i] = TypeTraits<T>::from_float(v[i] * scale);
        }
        if constexpr (kRunBytes % 16 == 0) {
            store_vec<uint4>(dst, rounded);
        } else if constexpr (kRunBytes % 8 == 0) {
            store_vec<uint64_t>(dst, rounded);
        } else if constexpr (kRunBytes % 4 == 0) {
            store_vec<uint32_t>(dst, rounded);
        } else {
            store_vec<uint16_t>(dst, rounded);
        }
    }

   private:
    template <typename VecT>
    static __device__ __forceinline__ void load_vec(const T* __restrict__ src, float (&v)[E]) {
        constexpr int kVecCount = kRunBytes / static_cast<int>(sizeof(VecT));
        VecT raw[kVecCount];
        const VecT* packed = reinterpret_cast<const VecT*>(src);
#pragma unroll
        for (int i = 0; i < kVecCount; ++i) {
            raw[i] = packed[i];
        }
        const T* elems = reinterpret_cast<const T*>(raw);
#pragma unroll
        for (int i = 0; i < E; ++i) {
            v[i] = TypeTraits<T>::to_float(elems[i]);
        }
    }

    template <typename VecT>
    static __device__ __forceinline__ void store_vec(T* __restrict__ dst, const T (&vals)[E]) {
        constexpr int kVecCount = kRunBytes / static_cast<int>(sizeof(VecT));
        VecT raw[kVecCount];
        T* elems = reinterpret_cast<T*>(raw);
#pragma unroll
        for (int i = 0; i < E; ++i) {
            elems[i] = vals[i];
        }
        VecT* packed = reinterpret_cast<VecT*>(dst);
#pragma unroll
        for (int i = 0; i < kVecCount; ++i) {
            packed[i] = raw[i];
        }
    }
};

// The butterfly itself, shared by the plain M2 kernel and the fused M4 kernel so that
// there is exactly one implementation of the transform arithmetic. A row of length d is
// owned by 2^kLogT lanes of the same warp, each holding kE = d / 2^kLogT contiguous
// elements. The stage with stride `len` pairs index i with i ^ len:
//   * len <  kE: both indices are in this thread's own registers -> plain adds;
//   * len >= kE: the partner is lane ^ (len / kE) at the same element index, so one
//     __shfl_xor_sync per element is enough, with the sign folded into the sum.
template <int kLogD, int kLogT>
struct Butterfly {
    static constexpr int kE = (1 << kLogD) / (1 << kLogT);

    static __device__ __forceinline__ void run(float (&v)[kE], int lane) {
        // Stages whose partner sits in this thread's own registers.
#pragma unroll
        for (int len = 1; len < kE; len <<= 1) {
#pragma unroll
            for (int i = 0; i < kE; ++i) {
                if ((i & len) == 0) {
                    const float a = v[i];
                    const float b = v[i + len];
                    v[i] = a + b;
                    v[i + len] = a - b;
                }
            }
        }

        // Stages whose partner is held by another thread of the same row.
        if constexpr (kLogT > 0) {
#pragma unroll
            for (int len = kE; len < (1 << kLogD); len <<= 1) {
                const int lane_mask = len / kE;
                const float sign = (lane & lane_mask) != 0 ? -1.0f : 1.0f;
#pragma unroll
                for (int i = 0; i < kE; ++i) {
                    const float other = __shfl_xor_sync(0xffffffffu, v[i], lane_mask, 1 << kLogT);
                    v[i] = sign * v[i] + other;
                }
            }
        }
    }
};

template <typename T, int kLogD, int kLogT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
    fwht_kernel(const T* __restrict__ x, T* __restrict__ y, long num_rows, float scale) {
    constexpr int kD = 1 << kLogD;
    constexpr int kT = 1 << kLogT;
    constexpr int kE = kD / kT;

    const int lane = threadIdx.x & (kT - 1);
    const int rows_per_block = blockDim.x >> kLogT;
    long row = static_cast<long>(blockIdx.x) * rows_per_block + (threadIdx.x >> kLogT);

    // Threads of the last block beyond num_rows keep running: their shuffle
    // partners have to stay active. They reuse the last row and skip the store.
    const bool owns_row = row < num_rows;
    if (!owns_row) {
        row = num_rows - 1;
    }

    float v[kE];
    RunIO<T, kE>::load(x + row * kD + lane * kE, v);
    Butterfly<kLogD, kLogT>::run(v, lane);

    if (owns_row) {
        RunIO<T, kE>::store(y + row * kD + lane * kE, v, scale);
    }
}

// Number of blocks to keep the machine busy, queried once and cached.
int target_block_count() {
    static const int cached = [] {
        int sm_count = 0;
        HW_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
        return std::max(1, sm_count) * kTargetBlocksPerSm;
    }();
    return cached;
}

// Block shape shared by the FWHT and the quantizer launchers: a few blocks per SM,
// whole warps, at or below kMaxThreadsPerBlock, and a multiple of the row's thread
// count so that every thread belongs to a row.
struct BlockShape {
    int threads;
    int rows_per_block;
};

BlockShape choose_block_shape(long rows, int threads_per_row) {
    const long wanted = std::max<long>(1, (rows + target_block_count() - 1) / target_block_count());
    const int max_rows_per_block = kMaxThreadsPerBlock / threads_per_row;
    const int wanted_rows_per_block = static_cast<int>(std::min<long>(wanted, max_rows_per_block));
    int threads = std::max(32, ((wanted_rows_per_block * threads_per_row + 31) / 32) * 32);
    threads = std::min(threads, kMaxThreadsPerBlock);
    return {threads, threads / threads_per_row};
}

template <typename T, int kLogD>
bool launch_fwht_typed(const void* x, void* y, long rows, float scale, cudaStream_t stream) {
    constexpr int kLogT = preferred_log_t(kLogD);
    constexpr int kT = 1 << kLogT;
    static_assert((1 << kLogD) % kT == 0, "threads per row must divide the row length");

    const BlockShape block = choose_block_shape(rows, kT);
    const long blocks = (rows + block.rows_per_block - 1) / block.rows_per_block;
    fwht_kernel<T, kLogD, kLogT>
        <<<static_cast<unsigned>(blocks), static_cast<unsigned>(block.threads), 0, stream>>>(
            reinterpret_cast<const T*>(x), reinterpret_cast<T*>(y), rows, scale);
    HW_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_fwht_dispatch(const void* x, void* y, int d, long rows, float scale,
                          cudaStream_t stream) {
    switch (d) {
        case 2:
            return launch_fwht_typed<T, 1>(x, y, rows, scale, stream);
        case 4:
            return launch_fwht_typed<T, 2>(x, y, rows, scale, stream);
        case 8:
            return launch_fwht_typed<T, 3>(x, y, rows, scale, stream);
        case 16:
            return launch_fwht_typed<T, 4>(x, y, rows, scale, stream);
        case 32:
            return launch_fwht_typed<T, 5>(x, y, rows, scale, stream);
        case 64:
            return launch_fwht_typed<T, 6>(x, y, rows, scale, stream);
        case 128:
            return launch_fwht_typed<T, 7>(x, y, rows, scale, stream);
        case 256:
            return launch_fwht_typed<T, 8>(x, y, rows, scale, stream);
        case 512:
            return launch_fwht_typed<T, 9>(x, y, rows, scale, stream);
        case 1024:
            return launch_fwht_typed<T, 10>(x, y, rows, scale, stream);
        default:
            return false;  // no instantiation for this row length
    }
}

// ---------------------------------------------------------------------------
// M3: the same transform expressed as a GEMM on Tensor Cores
//
// Y (M x d) = X (M x d) * H_d (d x d), computed as a tiled WMMA GEMM. Every entry
// of H_d is +-1, so the B operand is exact in fp16 and bf16 and the error budget
// stays M2's: FP32 accumulation, one rounding on the way out, 1/sqrt(d) folded into
// the epilogue.
//
// The d x d matrix is never materialised. Sylvester's construction satisfies
//
//   H_d[(i1,i0),(j1,j0)] = H_{d/16}[i1,j1] * H_16[i0,j0],   i = 16*i1 + i0,
//
// because H_n[i,j] = (-1)^popcount(i&j) and the bits of i1 and i0 do not overlap. So
// the B tile of a (k, n) tile pair is +H_16 or -H_16, selected by
// sign = (-1)^popcount(k & n): two 512 B tiles in shared memory, independent of d.
// Storing H_d outright would need 512 KiB at d = 512, well past the 100 KiB an SM
// has. (Checked against the recursive construction before writing this: the tile
// identity holds for every d = 16..1024 and the tiled accumulation reproduces the
// explicit matmul exactly.)
//
// Mapping: a block owns 16 rows and kCols = min(d, 256) output columns; each warp
// owns kCols / kWarps of those columns and runs the whole K = d loop in steps of 16.
// The row block is staged in shared memory and padded, because a leading dimension
// equal to d (a multiple of 64 elements) would pile all 16 rows of a fragment load
// onto the same banks.
//
// The epilogue has to pass through shared memory: a WMMA accumulator fragment holds
// float, store_matrix_sync can only write a buffer of the fragment's own type, and
// the output is fp16/bf16. Fragments land in fp32 shared memory, threads pick up
// their 16 B chunks from there, scale, round and write out; rows past the end of the
// tensor are simply not stored, which is how row counts smaller than 16 are handled.
// ---------------------------------------------------------------------------

constexpr int kTcTile = 16;
// More than one column block only when d exceeds this; 256 keeps the accumulator
// count per warp at 4 and the shared memory footprint small.
constexpr int kTcMaxColsPerBlock = 256;
constexpr int kTcMaxWarps = 4;
// Shared memory row pads, in elements. Padding is what keeps the rows of a fragment
// load off the same banks.
constexpr int kTcARowPad = 8;
constexpr int kTcEpiRowPad = 8;

// Used from the host launcher and from the kernel, so it needs both annotations (the
// same trap as the host-only TypeTraits conversion helpers below).
__host__ __device__ constexpr int tc_warps_for(int cols) {
    const int warps = cols / kTcTile;
    return warps < 1 ? 1 : (warps > kTcMaxWarps ? kTcMaxWarps : warps);
}

template <typename T, int kLogD>
__global__ void __launch_bounds__(kTcMaxWarps * 32)
    tc_fwht_kernel(const T* __restrict__ x, T* __restrict__ y, long num_rows, float scale) {
    constexpr int kD = 1 << kLogD;
    constexpr int kCols = kD < kTcMaxColsPerBlock ? kD : kTcMaxColsPerBlock;
    constexpr int kWarps = tc_warps_for(kCols);
    constexpr int kColsPerWarp = kCols / kWarps;
    constexpr int kNTilesPerWarp = kColsPerWarp / kTcTile;
    constexpr int kKTiles = kD / kTcTile;
    constexpr int kColBlocks = kD / kCols;
    constexpr int kARowStride = kD + kTcARowPad;
    constexpr int kEpiRowStride = kColsPerWarp + kTcEpiRowPad;
    constexpr int kVec = 8;  // 16 B, the widest vector for both element types
    static_assert(kColsPerWarp % kTcTile == 0, "the column split must stay tile aligned");
    static_assert(kCols * kColBlocks == kD, "the column blocks must cover the row");

    // The A staging and the fp32 epilogue staging are never live at the same time, so
    // they share one buffer; the __syncthreads() after the K loop makes that safe.
    constexpr int kABytes = kTcTile * kARowStride * static_cast<int>(sizeof(T));
    constexpr int kEpiBytes = kWarps * kTcTile * kEpiRowStride * static_cast<int>(sizeof(float));
    constexpr int kMainBytes = kABytes > kEpiBytes ? kABytes : kEpiBytes;

    __shared__ alignas(16) unsigned char s_main[kMainBytes];
    __shared__ T s_h16_pos[kTcTile * kTcTile];
    __shared__ T s_h16_neg[kTcTile * kTcTile];

    T* const s_a = reinterpret_cast<T*>(s_main);
    float* const s_epi = reinterpret_cast<float*>(s_main);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // +-H_16, with H_16[i][j] = (-1)^popcount(i & j). 256 entries and as few as 32
    // threads per block, so this has to be a strided loop and not a single guarded
    // store: writing only the first blockDim.x entries leaves the rest zero, which
    // silently turns the transform into a partial sum.
    for (int v = tid; v < kTcTile * kTcTile; v += blockDim.x) {
        const int i = v / kTcTile;
        const int j = v % kTcTile;
        const float s = ((__popc(i & j) & 1) == 0) ? 1.0f : -1.0f;
        s_h16_pos[v] = TypeTraits<T>::from_float(s);
        s_h16_neg[v] = TypeTraits<T>::from_float(-s);
    }
    __syncthreads();

    // B is the same two fragments for every tile pair, so both stay in registers and
    // B costs nothing after the prologue.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                           nvcuda::wmma::row_major>
        fb_pos;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                           nvcuda::wmma::row_major>
        fb_neg;
    nvcuda::wmma::load_matrix_sync(fb_pos, s_h16_pos, kTcTile);
    nvcuda::wmma::load_matrix_sync(fb_neg, s_h16_neg, kTcTile);

    // Stage the row block. Rows past the end of the tensor reuse the last real row;
    // they are excluded from the store at the bottom, so any row count works, including
    // fewer than 16 rows.
    const long row0 = static_cast<long>(blockIdx.x) * kTcTile;
    for (int v = tid; v < kTcTile * kD / kVec; v += blockDim.x) {
        const int row = (v * kVec) / kD;
        const int col = (v * kVec) % kD;
        const long src_row = row0 + row < num_rows ? row0 + row : num_rows - 1;
        reinterpret_cast<uint4*>(s_a + row * kARowStride + col)[0] =
            reinterpret_cast<const uint4*>(x + src_row * kD + col)[0];
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTcTile, kTcTile, kTcTile, float>
        acc[kNTilesPerWarp];
#pragma unroll
    for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
        nvcuda::wmma::fill_fragment(acc[nt], 0.0f);
    }

    const int first_n_tile = blockIdx.y * (kCols / kTcTile) + warp * kNTilesPerWarp;
// The K loop stays rolled: unroll factors 2 and 4 measured the same within noise,
// and at d = 1024 a full unroll is 64 fragment loads plus several MMAs each.
#pragma unroll 1
    for (int kt = 0; kt < kKTiles; ++kt) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTcTile, kTcTile, kTcTile, T,
                               nvcuda::wmma::row_major>
            fa;
        nvcuda::wmma::load_matrix_sync(fa, s_a + kt * kTcTile, kARowStride);
#pragma unroll
        for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
            const int n_tile = first_n_tile + nt;
            // Pick the fragment rather than branching on the sign. Branching made nvcc
            // predicate both HMMAs and insert a warp sync per mma in the fp16 build, so
            // the MMA issue slots doubled: the same kernel ran 1.5x slower in fp16 than
            // in bf16 until this became a select.
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                                   nvcuda::wmma::row_major>
                fb = ((__popc(kt & n_tile) & 1) == 0) ? fb_pos : fb_neg;
            nvcuda::wmma::mma_sync(acc[nt], fa, fb, acc[nt]);
        }
    }
    // Every warp has to be done reading the A staging before the epilogue reuses it.
    __syncthreads();

    float* const s_warp = s_epi + warp * (kTcTile * kEpiRowStride);
#pragma unroll
    for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
        nvcuda::wmma::store_matrix_sync(s_warp + nt * kTcTile, acc[nt], kEpiRowStride,
                                        nvcuda::wmma::mem_row_major);
    }
    // store_matrix_sync distributes the fragment over the lanes in its own layout, so
    // the reads below need the warp back in step.
    __syncwarp();

    constexpr int kVecsPerRow = kColsPerWarp / kVec;
    const int col0 = blockIdx.y * kCols + warp * kColsPerWarp;
#pragma unroll
    for (int v = lane; v < kTcTile * kVecsPerRow; v += 32) {
        const int row = v / kVecsPerRow;
        const int vec_in_row = v % kVecsPerRow;
        const long dst_row = row0 + row;
        if (dst_row >= num_rows) {
            continue;
        }
        const float* const src = s_warp + row * kEpiRowStride + vec_in_row * kVec;
        alignas(16) T out[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            out[i] = TypeTraits<T>::from_float(src[i] * scale);
        }
        reinterpret_cast<uint4*>(y + dst_row * kD + col0 + vec_in_row * kVec)[0] =
            reinterpret_cast<const uint4*>(out)[0];
    }
}

template <typename T, int kLogD>
bool launch_tc_typed(const void* x, void* y, long rows, float scale, cudaStream_t stream) {
    constexpr int kD = 1 << kLogD;
    constexpr int kCols = kD < kTcMaxColsPerBlock ? kD : kTcMaxColsPerBlock;
    constexpr int kWarps = tc_warps_for(kCols);
    const long row_blocks = (rows + kTcTile - 1) / kTcTile;
    const dim3 grid(static_cast<unsigned>(row_blocks), static_cast<unsigned>(kD / kCols));
    tc_fwht_kernel<T, kLogD><<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const T*>(x), reinterpret_cast<T*>(y), rows, scale);
    HW_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_tc_dispatch(const void* x, void* y, int d, long rows, float scale,
                        cudaStream_t stream) {
    switch (d) {
        case 16:
            return launch_tc_typed<T, 4>(x, y, rows, scale, stream);
        case 32:
            return launch_tc_typed<T, 5>(x, y, rows, scale, stream);
        case 64:
            return launch_tc_typed<T, 6>(x, y, rows, scale, stream);
        case 128:
            return launch_tc_typed<T, 7>(x, y, rows, scale, stream);
        case 256:
            return launch_tc_typed<T, 8>(x, y, rows, scale, stream);
        case 512:
            return launch_tc_typed<T, 9>(x, y, rows, scale, stream);
        case 1024:
            return launch_tc_typed<T, 10>(x, y, rows, scale, stream);
        default:
            return false;  // no instantiation for this row length
    }
}

// ---------------------------------------------------------------------------
// M4: FP8 E4M3 quantization, standalone and fused into an epilogue
//
// The rotated activation is quantized per token: one scale per row, mapping that row's
// largest magnitude onto 448, the largest E4M3 value. Rounding is the same
// __nv_cvt_float_to_fp8(__NV_SATFINITE, __NV_E4M3) call the host reference uses, and
// that call was checked to agree with the device instruction on every rounding
// boundary of the type before any of this was written.
//
// The fused kernels quantize the values the non-fused kernel would have written to
// memory: they round the accumulated row to the activation type first, take the row
// maximum of those rounded values, and convert from there. That is what makes
// "fused == transform then quantize" reproducible byte for byte. Quantizing the raw
// FP32 accumulator instead would differ by one code wherever the two accumulation
// orders happen to straddle a rounding boundary, and the whole point of fusing is that
// the quantized result must not depend on which kernel produced the rotation.
//
// The scale is a per-row quantity, so a block has to own whole rows. That is why the
// Tensor Core version does not split the columns across blocks the way the M3 kernel
// does, and why it stops at head_dim 512: 16 rows of FP32 staging for the epilogue is
// 16 * (d + pad) * 4 B, i.e. 33 KiB at d = 512 but 66 KiB at d = 1024, past the 48 KiB
// static shared memory limit. The butterfly version gets the row maximum from a shuffle
// over the lanes that share the row, so it covers every d the project supports.
// ---------------------------------------------------------------------------

// One thread's E output bytes. FP8 is a byte per element, so a run is 2/4/8/16/32 B and
// the widest vector that divides it is used.
template <int E>
__device__ __forceinline__ void store_fp8_run(uint8_t* __restrict__ dst, const uint8_t (&v)[E]) {
    if constexpr (E % 16 == 0) {
        for (int i = 0; i < E; i += 16) {
            reinterpret_cast<uint4*>(dst + i)[0] = reinterpret_cast<const uint4*>(v + i)[0];
        }
    } else if constexpr (E % 8 == 0) {
        for (int i = 0; i < E; i += 8) {
            reinterpret_cast<uint64_t*>(dst + i)[0] = reinterpret_cast<const uint64_t*>(v + i)[0];
        }
    } else if constexpr (E % 4 == 0) {
        for (int i = 0; i < E; i += 4) {
            reinterpret_cast<uint32_t*>(dst + i)[0] = reinterpret_cast<const uint32_t*>(v + i)[0];
        }
    } else {
        for (int i = 0; i < E; ++i) {
            dst[i] = v[i];
        }
    }
}

// Row maximum across the 2^kLogT lanes that share a row, so every lane ends up with it.
template <int kLogT>
__device__ __forceinline__ float row_amax_shfl(float local_max) {
    for (int mask = 1; mask < (1 << kLogT); mask <<= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask, 1 << kLogT));
    }
    return local_max;
}

// Quantizer only: the second half of the two-stage pipeline. Same thread mapping as the
// FWHT kernel (2^kLogT lanes per row, kE elements each), so the row maximum is a local
// reduction plus a shuffle butterfly.
template <typename T, int kLogD, int kLogT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
    quantize_fp8_kernel(const T* __restrict__ y, uint8_t* __restrict__ q,
                        float* __restrict__ scales, long num_rows) {
    constexpr int kD = 1 << kLogD;
    constexpr int kT = 1 << kLogT;
    constexpr int kE = kD / kT;

    const int lane = threadIdx.x & (kT - 1);
    const int rows_per_block = blockDim.x >> kLogT;
    long row = static_cast<long>(blockIdx.x) * rows_per_block + (threadIdx.x >> kLogT);
    const bool owns_row = row < num_rows;
    if (!owns_row) {
        row = num_rows - 1;
    }

    float v[kE];
    RunIO<T, kE>::load(y + row * kD + lane * kE, v);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kE; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = row_amax_shfl<kLogT>(amax);
    const float row_scale = fp8_e4m3_scale(amax);
    const float rscale = 1.0f / row_scale;

    alignas(16) uint8_t bytes[kE];
#pragma unroll
    for (int i = 0; i < kE; ++i) {
        bytes[i] = __nv_cvt_float_to_fp8(v[i] * rscale, __NV_SATFINITE, __NV_E4M3);
    }
    if (owns_row) {
        store_fp8_run<kE>(q + row * kD + lane * kE, bytes);
        if (lane == 0) {
            scales[row] = row_scale;
        }
    }
}

// Butterfly transform with the quantizer in its epilogue.
template <typename T, int kLogD, int kLogT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
    fwht_fp8_kernel(const T* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ scales,
                    long num_rows, float scale) {
    constexpr int kD = 1 << kLogD;
    constexpr int kT = 1 << kLogT;
    constexpr int kE = kD / kT;

    const int lane = threadIdx.x & (kT - 1);
    const int rows_per_block = blockDim.x >> kLogT;
    long row = static_cast<long>(blockIdx.x) * rows_per_block + (threadIdx.x >> kLogT);
    const bool owns_row = row < num_rows;
    if (!owns_row) {
        row = num_rows - 1;
    }

    float v[kE];
    RunIO<T, kE>::load(x + row * kD + lane * kE, v);
    Butterfly<kLogD, kLogT>::run(v, lane);

    // Same rounding as the non-fused kernel's store, then the row maximum of it.
    T rounded[kE];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kE; ++i) {
        rounded[i] = TypeTraits<T>::from_float(v[i] * scale);
        amax = fmaxf(amax, fabsf(TypeTraits<T>::to_float(rounded[i])));
    }
    amax = row_amax_shfl<kLogT>(amax);
    const float row_scale = fp8_e4m3_scale(amax);
    const float rscale = 1.0f / row_scale;

    alignas(16) uint8_t bytes[kE];
#pragma unroll
    for (int i = 0; i < kE; ++i) {
        bytes[i] = __nv_cvt_float_to_fp8(TypeTraits<T>::to_float(rounded[i]) * rscale,
                                         __NV_SATFINITE, __NV_E4M3);
    }
    if (owns_row) {
        store_fp8_run<kE>(q + row * kD + lane * kE, bytes);
        if (lane == 0) {
            scales[row] = row_scale;
        }
    }
}

template <typename T, int kLogD>
bool launch_quantize_typed(const void* y, uint8_t* q, float* scales, long rows,
                           cudaStream_t stream) {
    constexpr int kLogT = preferred_log_t(kLogD);
    constexpr int kT = 1 << kLogT;
    static_assert((1 << kLogD) % kT == 0, "threads per row must divide the row length");

    const BlockShape block = choose_block_shape(rows, kT);
    const long blocks = (rows + block.rows_per_block - 1) / block.rows_per_block;
    quantize_fp8_kernel<T, kLogD, kLogT>
        <<<static_cast<unsigned>(blocks), static_cast<unsigned>(block.threads), 0, stream>>>(
            reinterpret_cast<const T*>(y), q, scales, rows);
    HW_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_quantize_dispatch(const void* y, uint8_t* q, float* scales, int d, long rows,
                              cudaStream_t stream) {
    switch (d) {
        case 2:
            return launch_quantize_typed<T, 1>(y, q, scales, rows, stream);
        case 4:
            return launch_quantize_typed<T, 2>(y, q, scales, rows, stream);
        case 8:
            return launch_quantize_typed<T, 3>(y, q, scales, rows, stream);
        case 16:
            return launch_quantize_typed<T, 4>(y, q, scales, rows, stream);
        case 32:
            return launch_quantize_typed<T, 5>(y, q, scales, rows, stream);
        case 64:
            return launch_quantize_typed<T, 6>(y, q, scales, rows, stream);
        case 128:
            return launch_quantize_typed<T, 7>(y, q, scales, rows, stream);
        case 256:
            return launch_quantize_typed<T, 8>(y, q, scales, rows, stream);
        case 512:
            return launch_quantize_typed<T, 9>(y, q, scales, rows, stream);
        case 1024:
            return launch_quantize_typed<T, 10>(y, q, scales, rows, stream);
        default:
            return false;  // no instantiation for this row length
    }
}

template <typename T, int kLogD>
bool launch_fwht_fp8_typed(const void* x, uint8_t* q, float* scales, long rows, float scale,
                           cudaStream_t stream) {
    constexpr int kLogT = preferred_log_t(kLogD);
    constexpr int kT = 1 << kLogT;
    static_assert((1 << kLogD) % kT == 0, "threads per row must divide the row length");

    const BlockShape block = choose_block_shape(rows, kT);
    const long blocks = (rows + block.rows_per_block - 1) / block.rows_per_block;
    fwht_fp8_kernel<T, kLogD, kLogT>
        <<<static_cast<unsigned>(blocks), static_cast<unsigned>(block.threads), 0, stream>>>(
            reinterpret_cast<const T*>(x), q, scales, rows, scale);
    HW_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_fwht_fp8_dispatch(const void* x, uint8_t* q, float* scales, int d, long rows,
                              float scale, cudaStream_t stream) {
    switch (d) {
        case 2:
            return launch_fwht_fp8_typed<T, 1>(x, q, scales, rows, scale, stream);
        case 4:
            return launch_fwht_fp8_typed<T, 2>(x, q, scales, rows, scale, stream);
        case 8:
            return launch_fwht_fp8_typed<T, 3>(x, q, scales, rows, scale, stream);
        case 16:
            return launch_fwht_fp8_typed<T, 4>(x, q, scales, rows, scale, stream);
        case 32:
            return launch_fwht_fp8_typed<T, 5>(x, q, scales, rows, scale, stream);
        case 64:
            return launch_fwht_fp8_typed<T, 6>(x, q, scales, rows, scale, stream);
        case 128:
            return launch_fwht_fp8_typed<T, 7>(x, q, scales, rows, scale, stream);
        case 256:
            return launch_fwht_fp8_typed<T, 8>(x, q, scales, rows, scale, stream);
        case 512:
            return launch_fwht_fp8_typed<T, 9>(x, q, scales, rows, scale, stream);
        case 1024:
            return launch_fwht_fp8_typed<T, 10>(x, q, scales, rows, scale, stream);
        default:
            return false;  // no instantiation for this row length
    }
}

// Columns per warp in the fused Tensor Core kernel. A block owns the whole row, so more
// warps means fewer column tiles each: kWarps = min(8, d / 64).
constexpr int kTcFp8ColsPerWarp = 64;
constexpr int kTcFp8MaxWarps = 8;
// Biggest head_dim the fused Tensor Core kernel handles; see the section comment above.
constexpr int kTcFp8MaxDim = 512;

// d / 64 warps keeps 64 columns (4 tiles) per warp, but a single warp makes the per-row
// epilogue a serial chain of 16 reductions with only 8 lanes doing work at d = 64, and
// that costs more than the traffic fusion saves. Two warps at 32 columns each is the
// smallest split that hides it; d = 16 has to stay at one warp because the column count
// cannot drop below one tile.
__host__ __device__ constexpr int tc_fp8_warps_for(int d) {
    if (d < 2 * kTcTile) {
        return 1;
    }
    const int warps = d / kTcFp8ColsPerWarp;
    return warps < 2 ? 2 : (warps > kTcFp8MaxWarps ? kTcFp8MaxWarps : warps);
}

// Tensor Core transform with the quantizer in its epilogue. The GEMM itself (the +-H_16
// tiles, the A staging, the K loop) is the M3 kernel; what differs is that a block owns
// the whole row and that the epilogue runs three passes instead of one:
//   store   - fragments land in one shared FP32 tile, one column slice per warp;
//   reduce  - per-row |maximum| of the values rounded to the activation type;
//   write   - vectorised scale-and-convert of the same tile.
template <typename T, int kLogD>
__global__ void __launch_bounds__(kTcFp8MaxWarps * 32)
    tc_fwht_fp8_kernel(const T* __restrict__ x, uint8_t* __restrict__ q,
                       float* __restrict__ scales_out, long num_rows, float scale) {
    constexpr int kD = 1 << kLogD;
    constexpr int kWarps = tc_fp8_warps_for(kD);
    constexpr int kColsPerWarp = kD / kWarps;
    constexpr int kNTilesPerWarp = kColsPerWarp / kTcTile;
    constexpr int kKTiles = kD / kTcTile;
    constexpr int kARowStride = kD + kTcARowPad;
    constexpr int kEpiRowStride = kD + kTcEpiRowPad;
    constexpr int kVec = 8;  // 16 B of FP32 in, 8 B of FP8 out per thread
    constexpr int kVecsPerRow = kD / kVec;
    static_assert(kColsPerWarp % kTcTile == 0, "the column split must stay tile aligned");

    constexpr int kABytes = kTcTile * kARowStride * static_cast<int>(sizeof(T));
    constexpr int kTileBytes = kTcTile * kEpiRowStride * static_cast<int>(sizeof(float));
    constexpr int kMainBytes = kABytes > kTileBytes ? kABytes : kTileBytes;

    __shared__ alignas(16) unsigned char s_main[kMainBytes];
    __shared__ T s_h16_pos[kTcTile * kTcTile];
    __shared__ T s_h16_neg[kTcTile * kTcTile];
    __shared__ float s_amax[kTcTile];
    __shared__ float s_rscale[kTcTile];

    T* const s_a = reinterpret_cast<T*>(s_main);
    float* const s_tile = reinterpret_cast<float*>(s_main);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // +-H_16 as in M3: 256 entries and as few as 32 threads per block, so this has to be
    // a strided loop rather than a single guarded store.
    for (int v = tid; v < kTcTile * kTcTile; v += blockDim.x) {
        const int i = v / kTcTile;
        const int j = v % kTcTile;
        const float s = ((__popc(i & j) & 1) == 0) ? 1.0f : -1.0f;
        s_h16_pos[v] = TypeTraits<T>::from_float(s);
        s_h16_neg[v] = TypeTraits<T>::from_float(-s);
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                           nvcuda::wmma::row_major>
        fb_pos;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                           nvcuda::wmma::row_major>
        fb_neg;
    nvcuda::wmma::load_matrix_sync(fb_pos, s_h16_pos, kTcTile);
    nvcuda::wmma::load_matrix_sync(fb_neg, s_h16_neg, kTcTile);

    const long row0 = static_cast<long>(blockIdx.x) * kTcTile;
    for (int v = tid; v < kTcTile * kD / kVec; v += blockDim.x) {
        const int row = (v * kVec) / kD;
        const int col = (v * kVec) % kD;
        const long src_row = row0 + row < num_rows ? row0 + row : num_rows - 1;
        reinterpret_cast<uint4*>(s_a + row * kARowStride + col)[0] =
            reinterpret_cast<const uint4*>(x + src_row * kD + col)[0];
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTcTile, kTcTile, kTcTile, float>
        acc[kNTilesPerWarp];
#pragma unroll
    for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
        nvcuda::wmma::fill_fragment(acc[nt], 0.0f);
    }

#pragma unroll 1
    for (int kt = 0; kt < kKTiles; ++kt) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTcTile, kTcTile, kTcTile, T,
                               nvcuda::wmma::row_major>
            fa;
        nvcuda::wmma::load_matrix_sync(fa, s_a + kt * kTcTile, kARowStride);
#pragma unroll
        for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
            const int n_tile = warp * kNTilesPerWarp + nt;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTcTile, kTcTile, kTcTile, T,
                                   nvcuda::wmma::row_major>
                fb = ((__popc(kt & n_tile) & 1) == 0) ? fb_pos : fb_neg;
            nvcuda::wmma::mma_sync(acc[nt], fa, fb, acc[nt]);
        }
    }

    // The A staging is dead and the epilogue tile reuses its buffer, and every warp has
    // to have written its column slice before the row-wise reduction can read it.
    __syncthreads();
    for (int nt = 0; nt < kNTilesPerWarp; ++nt) {
        nvcuda::wmma::store_matrix_sync(s_tile + warp * kColsPerWarp + nt * kTcTile, acc[nt],
                                        kEpiRowStride, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // Per-row maximum, rows spread over the warps, 32 lanes sweeping a row each and a
    // shuffle butterfly to finish. Taken from the shared tile rather than from the
    // accumulator fragments on purpose: the mapping of fragment elements to matrix
    // entries is not part of the WMMA contract.
    for (int r = warp; r < kTcTile; r += kWarps) {
        const float* const row_ptr = s_tile + r * kEpiRowStride;
        float amax = 0.0f;
        for (int vi = lane; vi < kVecsPerRow; vi += 32) {
            const float* const src = row_ptr + vi * kVec;
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const T rounded = TypeTraits<T>::from_float(src[i] * scale);
                amax = fmaxf(amax, fabsf(TypeTraits<T>::to_float(rounded)));
            }
        }
        amax = row_amax_shfl<5>(amax);
        if (lane == 0) {
            s_amax[r] = amax;
        }
    }
    __syncthreads();
    if (tid < kTcTile) {
        s_rscale[tid] = 1.0f / fp8_e4m3_scale(s_amax[tid]);
    }
    __syncthreads();

    // Strided over the whole block rather than over one warp: every (row, vector) pair is
    // written exactly once, instead of once per warp.
    for (int v = tid; v < kTcTile * kVecsPerRow; v += blockDim.x) {
        const int row = v / kVecsPerRow;
        const int vec_in_row = v % kVecsPerRow;
        const long dst_row = row0 + row;
        if (dst_row >= num_rows) {
            continue;
        }
        const float* const src = s_tile + row * kEpiRowStride + vec_in_row * kVec;
        const float rscale = s_rscale[row];
        alignas(16) uint8_t bytes[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const T rounded = TypeTraits<T>::from_float(src[i] * scale);
            bytes[i] = __nv_cvt_float_to_fp8(TypeTraits<T>::to_float(rounded) * rscale,
                                             __NV_SATFINITE, __NV_E4M3);
        }
        store_fp8_run<kVec>(q + dst_row * kD + vec_in_row * kVec, bytes);
    }
    if (tid < kTcTile) {
        const long dst_row = row0 + tid;
        if (dst_row < num_rows) {
            scales_out[dst_row] = fp8_e4m3_scale(s_amax[tid]);
        }
    }
}

template <typename T, int kLogD>
bool launch_tc_fp8_typed(const void* x, uint8_t* q, float* scales, long rows, float scale,
                         cudaStream_t stream) {
    constexpr int kWarps = tc_fp8_warps_for(1 << kLogD);
    const long row_blocks = (rows + kTcTile - 1) / kTcTile;
    tc_fwht_fp8_kernel<T, kLogD><<<static_cast<unsigned>(row_blocks), kWarps * 32, 0, stream>>>(
        reinterpret_cast<const T*>(x), q, scales, rows, scale);
    HW_CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool launch_tc_fp8_dispatch(const void* x, uint8_t* q, float* scales, int d, long rows, float scale,
                            cudaStream_t stream) {
    switch (d) {
        case 16:
            return launch_tc_fp8_typed<T, 4>(x, q, scales, rows, scale, stream);
        case 32:
            return launch_tc_fp8_typed<T, 5>(x, q, scales, rows, scale, stream);
        case 64:
            return launch_tc_fp8_typed<T, 6>(x, q, scales, rows, scale, stream);
        case 128:
            return launch_tc_fp8_typed<T, 7>(x, q, scales, rows, scale, stream);
        case 256:
            return launch_tc_fp8_typed<T, 8>(x, q, scales, rows, scale, stream);
        case 512:
            return launch_tc_fp8_typed<T, 9>(x, q, scales, rows, scale, stream);
        default:
            return false;  // no instantiation for this row length
    }
}

}  // namespace

void launch_hello(int* out, int value, cudaStream_t stream) {
    hello_kernel<<<1, 32, 0, stream>>>(out, value);
    HW_CUDA_CHECK(cudaGetLastError());
}

bool launch_wmma_identity_probe(DType dtype, bool* correct, cudaStream_t stream) {
    if (!required_arch_ok(dtype)) {
        return false;
    }
    if (dtype == DType::kFp16) {
        return run_identity_probe<__half>(correct, stream);
    }
    return run_identity_probe<__nv_bfloat16>(correct, stream);
}

bool launch_fwht_baseline(const void* x, void* y, const Shape& shape, DType dtype,
                          cudaStream_t stream) {
    if (x == nullptr || y == nullptr || !shape.valid() || shape.rows() <= 0) {
        return false;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    return dtype == DType::kFp16
               ? launch_fwht_dispatch<__half>(x, y, shape.head_dim, shape.rows(), scale, stream)
               : launch_fwht_dispatch<__nv_bfloat16>(x, y, shape.head_dim, shape.rows(), scale,
                                                     stream);
}

bool launch_hadamard_tc(const void* x, void* y, const Shape& shape, DType dtype,
                        cudaStream_t stream) {
    if (x == nullptr || y == nullptr || !shape.valid() || shape.rows() <= 0) {
        return false;
    }
    // One m16n16k16 tile spans 16 columns, so head_dim < 16 cannot be tiled along K at
    // all. Those sizes go to the register butterfly kernel, which is already exact
    // there.
    if (shape.head_dim < kTcTile) {
        return launch_fwht_baseline(x, y, shape, dtype, stream);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    return dtype == DType::kFp16
               ? launch_tc_dispatch<__half>(x, y, shape.head_dim, shape.rows(), scale, stream)
               : launch_tc_dispatch<__nv_bfloat16>(x, y, shape.head_dim, shape.rows(), scale,
                                                   stream);
}

bool launch_quantize_fp8(const void* y, uint8_t* q, float* scales, const Shape& shape, DType dtype,
                         cudaStream_t stream) {
    if (y == nullptr || q == nullptr || scales == nullptr || !shape.valid() || shape.rows() <= 0) {
        return false;
    }
    return dtype == DType::kFp16 ? launch_quantize_dispatch<__half>(y, q, scales, shape.head_dim,
                                                                    shape.rows(), stream)
                                 : launch_quantize_dispatch<__nv_bfloat16>(
                                       y, q, scales, shape.head_dim, shape.rows(), stream);
}

bool launch_fwht_baseline_fp8(const void* x, uint8_t* q, float* scales, const Shape& shape,
                              DType dtype, cudaStream_t stream) {
    if (x == nullptr || q == nullptr || scales == nullptr || !shape.valid() || shape.rows() <= 0) {
        return false;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    return dtype == DType::kFp16 ? launch_fwht_fp8_dispatch<__half>(x, q, scales, shape.head_dim,
                                                                    shape.rows(), scale, stream)
                                 : launch_fwht_fp8_dispatch<__nv_bfloat16>(
                                       x, q, scales, shape.head_dim, shape.rows(), scale, stream);
}

bool launch_hadamard_tc_fp8(const void* x, uint8_t* q, float* scales, const Shape& shape,
                            DType dtype, cudaStream_t stream) {
    if (x == nullptr || q == nullptr || scales == nullptr || !shape.valid() || shape.rows() <= 0) {
        return false;
    }
    // Below one m16n16k16 tile there is nothing for the Tensor Cores to do; above
    // kTcFp8MaxDim the FP32 epilogue tile would not fit in static shared memory. Both
    // ends are covered by the butterfly quantizer, which shares this scale convention.
    if (shape.head_dim < kTcTile) {
        return launch_fwht_baseline_fp8(x, q, scales, shape, dtype, stream);
    }
    if (shape.head_dim > kTcFp8MaxDim) {
        return false;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    return dtype == DType::kFp16 ? launch_tc_fp8_dispatch<__half>(x, q, scales, shape.head_dim,
                                                                  shape.rows(), scale, stream)
                                 : launch_tc_fp8_dispatch<__nv_bfloat16>(
                                       x, q, scales, shape.head_dim, shape.rows(), scale, stream);
}

}  // namespace hadamard
