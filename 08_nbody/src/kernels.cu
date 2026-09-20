// kernels.cu —— 力计算 kernel（V3）与积分器 kernel
//
// V3 = shared memory 分块 + float4 打包 + rsqrtf。
// 每一层为什么这样做、收益来自哪里、实测拿到多少，见 kernels.cuh 的长注释；
// 完整的五档优化阶梯与逐档对比数据见开发版 n_version5 与 n_bodyversion.md §3.1。
#include "cuda_utils.cuh"
#include "kernels.cuh"

namespace nbody {

namespace {

// V3/V4 共用的单次交互。用 __forceinline__ 确保不产生函数调用开销。
//
// rsqrt 与 `1/sqrt` 的区别：
//   - `1/sqrt(x)`：一条 IEEE 正确舍入的 sqrt（多周期迭代）+ 一条除法。
//   - `rsqrtf(x)`：一条 SFU 指令，约 2^-22 相对误差（比 FP32 的 2^-24 略差）。
// 精度代价是真实存在的，所以这一档必须与朴素实现做能量误差对比后才能采纳，
// 不能只看快了多少。实测能量误差仍在 1e-5 量级（n_bodyversion.md §3.1）。
__device__ __forceinline__ void Interact(real dx, real dy, real dz, real mj,
                                         real G, real eps2, real *ax, real *ay,
                                         real *az) {
  const real r2 = dx * dx + dy * dy + dz * dz + eps2;
#ifdef NBODY_FP64
  // FP64 没有 rsqrt 的 SFU 指令，rsqrt(double) 由多条指令合成，
  // 相比 1/sqrt 无优势。所以 FP64 编译下这一档退化回 `1/sqrt`。
  // 这只影响 -DNBODY_FP64=ON 的小 N 验证构建，不影响主线性能。
  const real inv_r = real(1) / sqrt(r2);
#else
  const real inv_r = rsqrtf(r2);
#endif
  const real s = G * mj * inv_r * inv_r * inv_r;
  *ax += dx * s;
  *ay += dy * s;
  *az += dz * s;
}

// ===========================================================================
// V3：力计算的全部内容。
//
// 结构：把源粒子按 blockDim.x 切成 tile；每个 tile 由 block 协作载入
// shared memory，同步后所有线程遍历这 blockDim.x 个源粒子做交互。
// 每从 smem 读一个源粒子只做一次交互（未做线程粗化），交互用 rsqrtf。
//
// 四个容易写错的地方，逐一说明：
//
// 1. 两个 __syncthreads() 都必需。
//    第一个保证 smem 写入对全 block 可见；第二个（循环末）保证所有线程
//    用完当前 tile 才被下一轮覆写。少了第二个会出现"快线程覆写慢线程
//    还在读的数据"。症状是结果随 block 数与运行次数随机波动的 O(0.1) 误差
//    ——不崩溃、不报错、不总是错，极难查。
//    已实测：去掉这一行后跨 kernel 一致性检查的相对误差从位级重合跳到
//    2e-1 ~ 5e-1 且每次运行都不同。
//
// 2. 边界处理不能 early-return。
//    `if (i >= n) return;` 会让越界线程不参与 __syncthreads()，
//    导致同一 block 内活跃线程永远等不齐（死锁或未定义行为）。
//    正确做法：越界线程照常参与循环与同步，只是不写回结果。
//
// 3. 尾部 tile 不足 blockDim.x 个时，多余的 smem 槽填 mass=0 的哑元。
//    于是它们对加速度的贡献恰好为 0，不需要在内层加分支。
//    位置也要写 0（而非留旧值）：旧值可能让 r2 极小而放大 0*inf 风险。
//
// 4. 自交互（j == i）不需要分支跳过。
//    此时 dx=dy=dz=0，而 r2 = eps^2 > 0，分子为零向量，贡献恰好为 0。
//    省掉分支避免 warp 内发散。代价是 softening 必须 > 0
//    （否则 0/0 = NaN 会静默污染全部结果），main.cu 在启动时校验。
// ===========================================================================
__global__ void KernelComputeAccV3(const real4 *__restrict__ pos_mass,
                                   real *__restrict__ acc_x,
                                   real *__restrict__ acc_y,
                                   real *__restrict__ acc_z, int n, real G,
                                   real eps2) {
  // 动态 shared memory：一段 real4 数组，长度 blockDim.x。
  // 用 extern 而非固定大小数组，是为了让 block size 成为可调参数。
  extern __shared__ real4 smem4[];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = (i < n);
  // 越界线程读全 0（任意合法值即可），它不写回结果。
  const real4 pi = active ? pos_mass[i] : make_real4(0, 0, 0, 0);

  real ax = 0, ay = 0, az = 0;

  for (int tile = 0; tile < n; tile += blockDim.x) {
    const int src = tile + threadIdx.x;
    smem4[threadIdx.x] =
        (src < n) ? pos_mass[src] : make_real4(0, 0, 0, 0); // w=mass=0 → 贡献 0
    __syncthreads();

    const int len = static_cast<int>(blockDim.x);
    // 展开 4 次让多条独立的 rsqrt 重叠发射，掩盖 SFU 延迟。
#pragma unroll 4
    for (int k = 0; k < len; ++k) {
      const real4 pj = smem4[k];
      Interact(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z, pj.w, G, eps2, &ax, &ay,
               &az);
    }
    // 与循环开头那个同等必要，且更容易被漏掉：少了它，跑得快的线程
    // 会在慢线程还在读当前 tile 时就开始覆写 smem。见上面的说明 1。
    __syncthreads();
  }

  if (active) {
    acc_x[i] = ax;
    acc_y[i] = ay;
    acc_z[i] = az;
  }
}

// ---------------------------------------------------------------------------
// 把 SoA 的 pos_{x,y,z} + mass 打包成 real4，供力计算 kernel 使用。
//
// 每步都要重打包一次：积分器（KickDrift）更新的是 SoA 的 pos_*，
// 而 V3 的 kernel 读 real4。两种布局必须在每次力评估前对齐。
// 这是 O(N) 的开销，相比 O(N^2) 的力计算可忽略（N=65536 时 <0.1%）。
//
// 为什么不干脆让积分器也用 real4：那会让"权威副本"变成 real4，
// 而轨迹记录、诊断量、CPU 交叉验证全都按标量 SoA 取数据，
// 每一处都要多一层拆包。保留一次打包换取数据布局的单一权威副本。
// ---------------------------------------------------------------------------
__global__ void KernelPackPosMass(const real *__restrict__ pos_x,
                                  const real *__restrict__ pos_y,
                                  const real *__restrict__ pos_z,
                                  const real *__restrict__ mass,
                                  real4 *__restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = make_real4(pos_x[i], pos_y[i], pos_z[i], mass[i]);
}

// Leapfrog KDK 的前半部分：kick(半步) + drift(整步)。
// 合成一个 kernel 是因为两者都是 elementwise 的，
// 拆开会多一次 launch 和一轮读写 vel。
__global__ void
KernelKickDrift(real *__restrict__ pos_x, real *__restrict__ pos_y,
                real *__restrict__ pos_z, real *__restrict__ vel_x,
                real *__restrict__ vel_y, real *__restrict__ vel_z,
                const real *__restrict__ acc_x, const real *__restrict__ acc_y,
                const real *__restrict__ acc_z, int n, real dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const real half_dt = dt * real(0.5);
  const real vx = vel_x[i] + half_dt * acc_x[i];
  const real vy = vel_y[i] + half_dt * acc_y[i];
  const real vz = vel_z[i] + half_dt * acc_z[i];
  vel_x[i] = vx;
  vel_y[i] = vy;
  vel_z[i] = vz;
  pos_x[i] += dt * vx;
  pos_y[i] += dt * vy;
  pos_z[i] += dt * vz;
}

// Leapfrog KDK 的后半部分：用新位置处的加速度再 kick 半步。
__global__ void KernelKick(real *__restrict__ vel_x, real *__restrict__ vel_y,
                           real *__restrict__ vel_z,
                           const real *__restrict__ acc_x,
                           const real *__restrict__ acc_y,
                           const real *__restrict__ acc_z, int n,
                           real half_dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  vel_x[i] += half_dt * acc_x[i];
  vel_y[i] += half_dt * acc_y[i];
  vel_z[i] += half_dt * acc_z[i];
}

// 显式 Euler：x_{k+1} = x_k + v_k*dt，v_{k+1} = v_k + a_k*dt。
// 注意位置更新用的是**旧**速度——这才是一阶显式 Euler。
// 若先更新 v 再用新 v 更新 x，那是 symplectic Euler，
// 能量表现会好得多，就无法展示"一阶方法的能量单调漂移"这个对照效果。
__global__ void
KernelEulerUpdate(real *__restrict__ pos_x, real *__restrict__ pos_y,
                  real *__restrict__ pos_z, real *__restrict__ vel_x,
                  real *__restrict__ vel_y, real *__restrict__ vel_z,
                  const real *__restrict__ acc_x,
                  const real *__restrict__ acc_y,
                  const real *__restrict__ acc_z, int n, real dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  pos_x[i] += dt * vel_x[i];
  pos_y[i] += dt * vel_y[i];
  pos_z[i] += dt * vel_z[i];
  vel_x[i] += dt * acc_x[i];
  vel_y[i] += dt * acc_y[i];
  vel_z[i] += dt * acc_z[i];
}

// 把 SoA 的 pos_{x,y,z} 打包成 xyz 交错，供轨迹记录一次性拷回主机。
// 在设备侧打包而非拷回三段再在主机侧交错：三次小拷贝的延迟高于一次大拷贝。
__global__ void KernelPackPositions(const real *__restrict__ pos_x,
                                    const real *__restrict__ pos_y,
                                    const real *__restrict__ pos_z,
                                    float *__restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[3 * i + 0] = static_cast<float>(pos_x[i]);
  out[3 * i + 1] = static_cast<float>(pos_y[i]);
  out[3 * i + 2] = static_cast<float>(pos_z[i]);
}

// 轨迹打包的中间缓冲。设为文件级静态是为了避免每次记录都 malloc/free；
// 容量按需增长。单线程使用，无需加锁。
float *g_pack_buffer = nullptr;
int g_pack_capacity = 0;

inline int GridFor(int n, int block) { return (n + block - 1) / block; }

} // namespace

void AllocateDevice(DeviceBodies *d, int n) {
  d->n = n;
  const size_t bytes = static_cast<size_t>(n) * sizeof(real);
  CUDA_CHECK(cudaMalloc(&d->pos_x, bytes));
  CUDA_CHECK(cudaMalloc(&d->pos_y, bytes));
  CUDA_CHECK(cudaMalloc(&d->pos_z, bytes));
  CUDA_CHECK(cudaMalloc(&d->vel_x, bytes));
  CUDA_CHECK(cudaMalloc(&d->vel_y, bytes));
  CUDA_CHECK(cudaMalloc(&d->vel_z, bytes));
  CUDA_CHECK(cudaMalloc(&d->acc_x, bytes));
  CUDA_CHECK(cudaMalloc(&d->acc_y, bytes));
  CUDA_CHECK(cudaMalloc(&d->acc_z, bytes));
  CUDA_CHECK(cudaMalloc(&d->mass, bytes));
  // 力计算 kernel 需要的 real4 缓冲，按 n 分配即可（kernel 里用 i < n
  // 挡住越界）。
  CUDA_CHECK(cudaMalloc(&d->pos_mass, static_cast<size_t>(n) * sizeof(real4)));
}

void FreeDevice(DeviceBodies *d) {
  cudaFree(d->pos_x);
  cudaFree(d->pos_y);
  cudaFree(d->pos_z);
  cudaFree(d->vel_x);
  cudaFree(d->vel_y);
  cudaFree(d->vel_z);
  cudaFree(d->acc_x);
  cudaFree(d->acc_y);
  cudaFree(d->acc_z);
  cudaFree(d->mass);
  cudaFree(d->pos_mass);
  *d = DeviceBodies{};
  if (g_pack_buffer) {
    cudaFree(g_pack_buffer);
    g_pack_buffer = nullptr;
    g_pack_capacity = 0;
  }
}

void UploadBodies(DeviceBodies *d, const HostBodies &h) {
  const size_t bytes = h.size() * sizeof(real);
  CUDA_CHECK(
      cudaMemcpy(d->pos_x, h.pos_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d->pos_y, h.pos_y.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d->pos_z, h.pos_z.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d->vel_x, h.vel_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d->vel_y, h.vel_y.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d->vel_z, h.vel_z.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d->mass, h.mass.data(), bytes, cudaMemcpyHostToDevice));
}

void DownloadBodies(HostBodies *h, const DeviceBodies &d) {
  h->resize(static_cast<size_t>(d.n));
  const size_t bytes = static_cast<size_t>(d.n) * sizeof(real);
  CUDA_CHECK(
      cudaMemcpy(h->pos_x.data(), d.pos_x, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h->pos_y.data(), d.pos_y, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h->pos_z.data(), d.pos_z, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h->vel_x.data(), d.vel_x, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h->vel_y.data(), d.vel_y, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h->vel_z.data(), d.vel_z, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h->mass.data(), d.mass, bytes, cudaMemcpyDeviceToHost));
}

void LaunchPackPositions(const DeviceBodies &d, float *dst,
                         cudaStream_t stream) {
  const int block = 256;
  KernelPackPositions<<<GridFor(d.n, block), block, 0, stream>>>(
      d.pos_x, d.pos_y, d.pos_z, dst, d.n);
  CUDA_CHECK_LAST();
}

void DownloadPositionsInterleaved(float *dst_xyz, const DeviceBodies &d) {
  const int needed = d.n * 3;
  if (g_pack_capacity < needed) {
    if (g_pack_buffer)
      CUDA_CHECK(cudaFree(g_pack_buffer));
    CUDA_CHECK(cudaMalloc(&g_pack_buffer,
                          static_cast<size_t>(needed) * sizeof(float)));
    g_pack_capacity = needed;
  }
  const int block = 256;
  KernelPackPositions<<<GridFor(d.n, block), block>>>(d.pos_x, d.pos_y, d.pos_z,
                                                      g_pack_buffer, d.n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(dst_xyz, g_pack_buffer,
                        static_cast<size_t>(needed) * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

size_t SharedBytesPerBlock(int block_size) {
  // 一段 real4 数组：x, y, z, mass 四个分量各占一个 real。
  return static_cast<size_t>(block_size) * sizeof(real4);
}

void LaunchPackPosMass(const DeviceBodies &d, cudaStream_t stream) {
  const int block = 256;
  KernelPackPosMass<<<GridFor(d.n, block), block, 0, stream>>>(
      d.pos_x, d.pos_y, d.pos_z, d.mass, d.pos_mass, d.n);
  CUDA_CHECK_LAST();
}

void LaunchComputeAcc(const DeviceBodies &d, real G, real softening,
                      int block_size, cudaStream_t stream) {
  const real eps2 = softening * softening;
  const size_t smem = SharedBytesPerBlock(block_size);

  // 力计算 kernel 读 real4，必须先把当前 SoA 位置打包过去。
  LaunchPackPosMass(d, stream);

  KernelComputeAccV3<<<GridFor(d.n, block_size), block_size, smem, stream>>>(
      d.pos_mass, d.acc_x, d.acc_y, d.acc_z, d.n, G, eps2);
  CUDA_CHECK_LAST();
}

void LaunchKickDrift(const DeviceBodies &d, real dt, int block_size,
                     cudaStream_t stream) {
  KernelKickDrift<<<GridFor(d.n, block_size), block_size, 0, stream>>>(
      d.pos_x, d.pos_y, d.pos_z, d.vel_x, d.vel_y, d.vel_z, d.acc_x, d.acc_y,
      d.acc_z, d.n, dt);
  CUDA_CHECK_LAST();
}

void LaunchKick(const DeviceBodies &d, real half_dt, int block_size,
                cudaStream_t stream) {
  KernelKick<<<GridFor(d.n, block_size), block_size, 0, stream>>>(
      d.vel_x, d.vel_y, d.vel_z, d.acc_x, d.acc_y, d.acc_z, d.n, half_dt);
  CUDA_CHECK_LAST();
}

void LaunchEulerUpdate(const DeviceBodies &d, real dt, int block_size,
                       cudaStream_t stream) {
  KernelEulerUpdate<<<GridFor(d.n, block_size), block_size, 0, stream>>>(
      d.pos_x, d.pos_y, d.pos_z, d.vel_x, d.vel_y, d.vel_z, d.acc_x, d.acc_y,
      d.acc_z, d.n, dt);
  CUDA_CHECK_LAST();
}

} // namespace nbody
