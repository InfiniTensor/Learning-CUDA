// diagnostics_gpu.cu —— GPU 诊断量的实现（FP64 / FP32 双精度档）
//
// 把 kernel 按累加器类型模板化，于是可以用**同一份代码**跑 double 与 float
// 两档。这不是为了给 FP32 找用途——FP32 档是**刻意的反面对照**，
// 用来实测"若尺子本身用 FP32 会怎样"。
//
// 为什么值得专门做这件事：从项目第一版起，"诊断量必须用 double 累加"
// 这个论断在四个版本的文档里反复出现，依据始终是一段纸面推导
// （相对误差 ~ eps_fp32 * sqrt(累加次数)）。推导可能对，也可能漏了因子——
// 而它恰恰是"能量守恒误差在合理范围内"这个硬性验收项的地基。
// 地基是推导还是实测，差别很大。见 n_bodyversion.md 附录 E。
//
// 选择哪一档由 ComputeDiagnosticsGpu 的 DiagAccum 参数决定，
// 主机侧默认走 kFloat64（正式数据只用这一档）。
#include <cmath>
#include <cstring>

#include "cuda_utils.cuh"
#include "diagnostics_gpu.cuh"

namespace nbody {
namespace {

// 全局累加器的槽位。用一个数组而非 8 个独立指针：
// 一次 cudaMemset 清零、一次 cudaMemcpy 取回，少 7 次 API 往返。
enum AccumSlot {
  kKinetic = 0,
  kPotential,
  kMomX,
  kMomY,
  kMomZ,
  kAngX,
  kAngY,
  kAngZ,
  kAccumSlots,
};

// ---------------------------------------------------------------------------
// double 的 atomicMax：CUDA 没有提供，用位模式序的性质自己做。
//
// IEEE-754 的一个性质：对**非负**浮点数，把位模式当作无符号整数比较，
// 得到的序与数值序一致（指数在高位、尾数在低位，都是无符号编码）。
// max|a| 恒为非负，所以可以直接对 bit pattern 做 atomicMax。
// 这个技巧对负数会失效（符号位使序反转），只能用在保证非负的量上。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void AtomicMaxNonNegDouble(unsigned long long *addr,
                                                      double value) {
  atomicMax(addr, static_cast<unsigned long long>(__double_as_longlong(value)));
}

// 主机侧把位模式还原成 double。没有 __longlong_as_double 的主机版本，
// 而 reinterpret_cast<double*> 是严格别名违规（UB，-O2 下真的会出错）。
// std::memcpy 是标准认可的位重解释方式，编译器会优化成一条 mov。
inline double BitsToDouble(unsigned long long bits) {
  double out = 0.0;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// ---------------------------------------------------------------------------
// 倒数平方根：**两档都用"正确舍入的 sqrt + 除法"，不用 rsqrt 近似。**
//
// 这一点是本实验能成立的关键。若 FP32 档用 rsqrtf（SFU 近似，~2^-22 相对
// 误差），那 FP32 与 FP64 就差了**两件事**：累加器类型 + 倒数平方根算法，
// 测出的误差无法归因到哪一件。用 1/sqrt 则两档的舍入规则完全相同
// （IEEE 正确舍入），**唯一的变量就是类型宽度**——这才是干净的对照。
//
// 顺带的好处：FP64 档从此与主机侧参考实现（CPU 用 1.0/std::sqrt）
// 使用完全相同的表达式，两者的一致性检查更严格。
// 代价是 FP64 档比 rsqrt 版略慢，但诊断是低频调用，不在乎。
// ---------------------------------------------------------------------------
__device__ __forceinline__ double InvSqrt(double x) { return 1.0 / sqrt(x); }
__device__ __forceinline__ float InvSqrt(float x) { return 1.0f / sqrtf(x); }

// 模板化的动态 shared memory。extern __shared__ 只能声明一次类型，
// 所以按 CUDA 的标准做法声明成字节数组再重解释。
template <typename T> __device__ __forceinline__ T *DynamicSmem() {
  extern __shared__ unsigned char raw_smem[];
  return reinterpret_cast<T *>(raw_smem);
}

// block 内树形归约。返回值只在 threadIdx.x == 0 上有效。
template <typename Acc> __device__ Acc BlockReduceSum(Acc *scratch, Acc value) {
  const int tid = static_cast<int>(threadIdx.x);
  scratch[tid] = value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
       stride >>= 1) {
    if (tid < stride)
      scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  return scratch[0];
}

// ---------------------------------------------------------------------------
// 诊断量 kernel。Acc = double（默认，正式用）或 float（反面对照）。
//
// 三处刻意的设计（与力计算 kernel 不同）：
//
//   1. **内层累加用 Acc**。这正是本实验的变量。
//
//   2. **自交互必须显式跳过 (j == i continue)**，不能沿用力计算那套
//      "靠 softening 让贡献自然为零"的技巧。力的分子是 (r_j - r_i)，
//      j==i 时为零向量所以贡献恰好为 0；但**势能的分子是常数 m_i*m_j**，
//      j==i 会贡献 -G*m_i^2/eps —— 一个巨大的伪项。这个坑很隐蔽：
//      力算对了、能量却系统性偏负，且 N 越大偏得越多。
//
//   3. **势能按 j != i 全求和后折半**，而非只统计 j > i。
//      后者会让线程工作量随 i 线性递减，warp 内负载严重不均。
//
// 注意全局 atomicAdd 始终是 double，即便 Acc=float：
// 每线程的内层链有 N 次累加，而全局只有 grid（=N/block）次，
// 前者比后者多两个数量级，误差由前者主导。把全局那一层固定成 double
// 能让实验干净地归因到"内层累加的类型"，而不是混进归约层的差异。
// ---------------------------------------------------------------------------
template <typename Acc>
__global__ void KernelDiagnostics(
    const real *__restrict__ pos_x, const real *__restrict__ pos_y,
    const real *__restrict__ pos_z, const real *__restrict__ vel_x,
    const real *__restrict__ vel_y, const real *__restrict__ vel_z,
    const real *__restrict__ mass, int n, double G_in, double eps2_in,
    double *__restrict__ accum, unsigned long long *__restrict__ max_acc2) {
  Acc *scratch = DynamicSmem<Acc>();

  const Acc G = static_cast<Acc>(G_in);
  const Acc eps2 = static_cast<Acc>(eps2_in);

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = (i < n);

  // 越界线程贡献 0，但仍要参与 __syncthreads()（归约里有同步）。
  Acc kin = 0, pot = 0;
  Acc mx = 0, my = 0, mz = 0;
  Acc ax_l = 0, ay_l = 0, az_l = 0;
  Acc acc2 = 0;

  if (active) {
    const Acc xi = static_cast<Acc>(pos_x[i]);
    const Acc yi = static_cast<Acc>(pos_y[i]);
    const Acc zi = static_cast<Acc>(pos_z[i]);
    const Acc vxi = static_cast<Acc>(vel_x[i]);
    const Acc vyi = static_cast<Acc>(vel_y[i]);
    const Acc vzi = static_cast<Acc>(vel_z[i]);
    const Acc mi = static_cast<Acc>(mass[i]);

    kin = Acc(0.5) * mi * (vxi * vxi + vyi * vyi + vzi * vzi);
    mx = mi * vxi;
    my = mi * vyi;
    mz = mi * vzi;
    // L = r x (m v)
    ax_l = mi * (yi * vzi - zi * vyi);
    ay_l = mi * (zi * vxi - xi * vzi);
    az_l = mi * (xi * vyi - yi * vxi);

    Acc gx = 0, gy = 0, gz = 0; // 加速度
    for (int j = 0; j < n; ++j) {
      if (j == i)
        continue; // 见上面第 2 点
      const Acc dx = static_cast<Acc>(pos_x[j]) - xi;
      const Acc dy = static_cast<Acc>(pos_y[j]) - yi;
      const Acc dz = static_cast<Acc>(pos_z[j]) - zi;
      const Acc r2 = dx * dx + dy * dy + dz * dz + eps2;
      const Acc inv_r = InvSqrt(r2);
      const Acc mj = static_cast<Acc>(mass[j]);
      pot -= G * mi * mj * inv_r; // 全求和，稍后折半
      const Acc s = G * mj * inv_r * inv_r * inv_r;
      gx += dx * s;
      gy += dy * s;
      gz += dz * s;
    }
    acc2 = gx * gx + gy * gy + gz * gz;
  }

  // 八个量各做一次 block 归约，再由 0 号线程 atomicAdd 到全局。
  // 先块内归约再 atomic：直接让每线程 atomicAdd 会产生 N 次争用，
  // 块内归约后全局原子操作降到 grid 次。
  const Acc sk = BlockReduceSum(scratch, kin);
  __syncthreads();
  const Acc sp = BlockReduceSum(scratch, pot);
  __syncthreads();
  const Acc smx = BlockReduceSum(scratch, mx);
  __syncthreads();
  const Acc smy = BlockReduceSum(scratch, my);
  __syncthreads();
  const Acc smz = BlockReduceSum(scratch, mz);
  __syncthreads();
  const Acc sax = BlockReduceSum(scratch, ax_l);
  __syncthreads();
  const Acc say = BlockReduceSum(scratch, ay_l);
  __syncthreads();
  const Acc saz = BlockReduceSum(scratch, az_l);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(&accum[kKinetic], static_cast<double>(sk));
    atomicAdd(&accum[kPotential], static_cast<double>(sp));
    atomicAdd(&accum[kMomX], static_cast<double>(smx));
    atomicAdd(&accum[kMomY], static_cast<double>(smy));
    atomicAdd(&accum[kMomZ], static_cast<double>(smz));
    atomicAdd(&accum[kAngX], static_cast<double>(sax));
    atomicAdd(&accum[kAngY], static_cast<double>(say));
    atomicAdd(&accum[kAngZ], static_cast<double>(saz));
  }

  // max|a|^2 用位模式 atomicMax（模长平方恒非负）。
  // 每线程各打一次：max 的争用远比 add 轻，不值得再做一轮块内归约。
  if (active)
    AtomicMaxNonNegDouble(max_acc2, static_cast<double>(acc2));
}

} // namespace

const char *DiagAccumName(DiagAccum a) {
  return a == DiagAccum::kFloat32 ? "fp32" : "fp64";
}

void AllocateDiagnosticsGpu(DiagnosticsGpuWorkspace *ws) {
  if (ws->allocated)
    return;
  CUDA_CHECK(cudaMalloc(&ws->accum, sizeof(double) * kAccumSlots));
  CUDA_CHECK(cudaMalloc(&ws->max_acc, sizeof(unsigned long long)));
  ws->allocated = true;
}

void FreeDiagnosticsGpu(DiagnosticsGpuWorkspace *ws) {
  if (!ws->allocated)
    return;
  cudaFree(ws->accum);
  cudaFree(ws->max_acc);
  *ws = DiagnosticsGpuWorkspace{};
}

Diagnostics ComputeDiagnosticsGpu(const DeviceBodies &d, real G, real softening,
                                  DiagnosticsGpuWorkspace *ws, int block_size,
                                  DiagAccum accum_kind, cudaStream_t stream) {
  AllocateDiagnosticsGpu(ws);

  // 每次调用前清零。max|a|^2 清成 0（模长平方的下界），
  // 位模式 atomicMax 的语义要求初值是合法的非负 double。
  CUDA_CHECK(
      cudaMemsetAsync(ws->accum, 0, sizeof(double) * kAccumSlots, stream));
  CUDA_CHECK(
      cudaMemsetAsync(ws->max_acc, 0, sizeof(unsigned long long), stream));

  const int grid = (d.n + block_size - 1) / block_size;
  const double g = static_cast<double>(G);
  const double eps2 =
      static_cast<double>(softening) * static_cast<double>(softening);

  if (accum_kind == DiagAccum::kFloat32) {
    const size_t smem = static_cast<size_t>(block_size) * sizeof(float);
    KernelDiagnostics<float><<<grid, block_size, smem, stream>>>(
        d.pos_x, d.pos_y, d.pos_z, d.vel_x, d.vel_y, d.vel_z, d.mass, d.n, g,
        eps2, ws->accum, ws->max_acc);
  } else {
    const size_t smem = static_cast<size_t>(block_size) * sizeof(double);
    KernelDiagnostics<double><<<grid, block_size, smem, stream>>>(
        d.pos_x, d.pos_y, d.pos_z, d.vel_x, d.vel_y, d.vel_z, d.mass, d.n, g,
        eps2, ws->accum, ws->max_acc);
  }
  CUDA_CHECK_LAST();

  double h[kAccumSlots] = {};
  unsigned long long h_max = 0;
  CUDA_CHECK(
      cudaMemcpyAsync(h, ws->accum, sizeof(h), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&h_max, ws->max_acc, sizeof(h_max),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Diagnostics out;
  out.kinetic = h[kKinetic];
  // 折半：kernel 里按 j != i 全求和，每对被数了两次。
  out.potential = h[kPotential] * 0.5;
  out.total_energy = out.kinetic + out.potential;
  out.momentum[0] = h[kMomX];
  out.momentum[1] = h[kMomY];
  out.momentum[2] = h[kMomZ];
  out.angular_momentum[0] = h[kAngX];
  out.angular_momentum[1] = h[kAngY];
  out.angular_momentum[2] = h[kAngZ];
  out.momentum_magnitude = std::sqrt(out.momentum[0] * out.momentum[0] +
                                     out.momentum[1] * out.momentum[1] +
                                     out.momentum[2] * out.momentum[2]);
  out.angular_momentum_magnitude =
      std::sqrt(out.angular_momentum[0] * out.angular_momentum[0] +
                out.angular_momentum[1] * out.angular_momentum[1] +
                out.angular_momentum[2] * out.angular_momentum[2]);
  out.max_acc_magnitude = std::sqrt(BitsToDouble(h_max));
  return out;
}

} // namespace nbody
