// =============================================================================
//  probe.cuh -- kernels behind the bandwidth roof measurement.
//
//  `--mode info` reports two numbers that are supposed to bound every kernel in
//  this project:
//
//    * the driver's theoretical DRAM bandwidth, from the memory clock and the
//      bus width, and
//    * the throughput of a DeviceToDevice copy.
//
//  Neither is a reliable roof.  On the MTT S4000 / MUSA 3.1 the *theoretical*
//  figure comes out at 512.8 GB/s while a plain D2D memcpy sustains 690 GB/s --
//  the driver's arithmetic is not a bound at all.  And the copy path may be a
//  different engine from the SIMT pipeline the kernels run on, so it can be
//  either above or below what a kernel can reach.  Normalising against the wrong
//  roof is how a report ends up claiming 96% of a peak that was never the peak.
//
//  So the roof is measured with the same execution resources the transform
//  kernels use: a grid-stride SIMT kernel with 128-bit accesses, split into the
//  three numbers that matter for a streaming operator (see RoofResult).
// =============================================================================
#pragma once

#include "fhwt/common.cuh"

namespace fhwt {

// Fills the buffers with a cheap pseudorandom pattern.  A constant pattern is
// what a memset would leave behind, and on parts with memory compression that
// makes the copy path look faster than any real workload could ever go -- the
// roof would then be an artefact of the test data rather than of the memory
// system.  The transform's input is random, so the roof's input is random too.
__global__ void roof_fill_kernel(uint4 *__restrict__ out, long long n4, unsigned seed) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    unsigned x = (unsigned)i * 2654435761u + seed;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    out[i] = make_uint4(x, x * 3u + 1u, x ^ 0x9e3779b9u, x + seed);
  }
}

// out[i] = in[i] with 128-bit accesses: read + write.  Four independent loads are
// issued per iteration so that the loop carries four outstanding requests per
// thread; without that the roof comes out a few percent low and the transform
// kernel looks like it beats a pure copy.
__global__ void roof_copy_kernel(const uint4 *__restrict__ in, uint4 *__restrict__ out,
                                 long long n4) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n4; i += 4 * stride) {
    const uint4 v0 = in[i];
    const uint4 v1 = in[i + stride];
    const uint4 v2 = in[i + 2 * stride];
    const uint4 v3 = in[i + 3 * stride];
    out[i] = v0;
    out[i + stride] = v1;
    out[i + 2 * stride] = v2;
    out[i + 3 * stride] = v3;
  }
  for (; i < n4; i += stride)
    out[i] = in[i];
}

// The same copy, repeated `reps` times inside one launch.  Needed for the
// L2-resident measurement: a small copy takes less time than a launch costs on
// a MUSA part (~20 us per launch versus ~30 us for copying 6 MiB), so timing one
// copy per launch measures the launch, not the memory system.
__global__ void roof_copy_loop_kernel(const uint4 *__restrict__ in, uint4 *__restrict__ out,
                                      long long n4, int reps) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  const long long i0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (int r = 0; r < reps; ++r) {
    for (long long i = i0; i < n4; i += stride)
      out[i] = in[i];
  }
}

// Read only.  The guard is never true for the test pattern, but the compiler
// cannot know that, so the loads survive.
__global__ void roof_read_kernel(const uint4 *__restrict__ in, unsigned *__restrict__ sink,
                                 long long n4) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const uint4 v = in[i];
    acc += (v.x ^ v.y) + (v.z ^ v.w);
  }
  if (acc == 0xdeadbeefu)
    sink[blockIdx.x] = acc;
}

// Write only.
__global__ void roof_write_kernel(uint4 *__restrict__ out, long long n4) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  const uint4 v = make_uint4(0x5a5a5a5au, 0x3c3c3c3cu, 0x0f0f0f0fu, 0xffffffffu);
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    out[i] = v;
  }
}

} // namespace fhwt
