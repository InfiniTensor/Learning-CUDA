#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
// 将表放入到常量内存加快速度
__constant__ float NF4_LUT[16] = {
    -1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
    -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
    0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
    0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f};
__constant__ __half NF4_LUT_HALF[16];
__constant__ __half CODE2_LUT[256];

//     Shape  Block   bnb (ms)  cuda (ms)     Speedup        MAE
//   102x102     64 0.06684775 0.00413914 16.15017009 0.00003332
//   102x102    128 0.07174208 0.00288320 24.88279773 0.00002438
//   512x768     64 0.07281434 0.00941510  7.73377961 0.00002742
//   512x768    128 0.06949299 0.00926707  7.49891559 0.00001532
// 1024x1024     64 0.06589485 0.01247328  5.28288084 0.00001693
// 1024x1024    128 0.06430317 0.00821562  7.82694414 0.00002801
// 2048x1536     64 0.06699008 0.02200211  3.04471150 0.00001925
// 2048x1536    128 0.07256377 0.02765293  2.62409014 0.00002384
// 4096x4096     64 0.07029203 0.14870676  0.47268888 0.00002044
// 4096x4096    128 0.06603654 0.09682874  0.68199326 0.00001764
__global__ void nf4_dequant_v1(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2, // 修改为 __half
    const __half *__restrict__ code2,   // 修改为 __half
    float offset, int64_t total_half_elements, int blocksize, int group_size,
    __half2 *__restrict__ output) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_half_elements)
    return;

  uint8_t val = packed[tid];

  int64_t idx_in_elements = tid << 1;
  int64_t block_idx = idx_in_elements / blocksize;
  int64_t group_idx = block_idx / group_size;

  // 3. 从 __half 转换为 float 进行高精度计算
  float s1 = __half2float(code2[absmax_q[block_idx]]);
  float s2 = __half2float(absmax2[group_idx]);
  float scale = (s1 * s2) + offset;

  // 4. 解包并应用 scale
  float v1 = NF4_LUT[val >> 4] * scale;
  float v2 = NF4_LUT[val & 0x0F] * scale;

  // 5. 写回
  output[tid] = __floats2half2_rn(v1, v2);
}

//     Shape  Block   bnb (ms)  cuda (ms)     Speedup        MAE
//   102x102     64 0.06894669 0.00526758 13.08886347 0.00021207
//   102x102    128 0.06629478 0.00492973 13.44795983 0.00026774
//   512x768     64 0.06693267 0.00527808 12.68125355 0.00019455
//   512x768    128 0.07099091 0.00955059  7.43314239 0.00020897
// 1024x1024     64 0.06750003 0.01292397  5.22285639 0.00017190
// 1024x1024    128 0.06577235 0.01272864  5.16727287 0.00028586
// 2048x1536     64 0.06630151 0.03046714  2.17616468 0.00017166
// 2048x1536    128 0.06532160 0.02048320  3.18903304 0.00027943
// 4096x4096     64 0.07393484 0.15142413  0.48826329 0.00022972
// 4096x4096    128 0.06733184 0.14666502  0.45908587 0.00029182
// 采用
__global__ void nf4_dequant_v2(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2,
    const __half *__restrict__ code2,
    float offset,
    int64_t total_half_elements,
    int blocksize,
    int group_size,
    __half2 *__restrict__ output)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_half_elements) return;

    uint8_t val = packed[tid];

    int64_t idx = tid << 1;
    int64_t block_idx = idx / blocksize;
    int64_t group_idx = block_idx / group_size;

    // ===== 全half scale =====
    __half s1 = code2[absmax_q[block_idx]];
    __half s2 = absmax2[group_idx];

    __half scale = __hadd(__hmul(s1, s2),__float2half(offset) );

    // LUT
    __half v1 = __hmul(__float2half(NF4_LUT[val >> 4]), scale);
    __half v2 = __hmul(__float2half(NF4_LUT[val & 0x0F]), scale);

    output[tid] = __halves2half2(v1, v2);
}


//     Shape  Block   bnb (ms)  cuda (ms)     Speedup        MAE
//   102x102     64 0.07130957 0.00292614 24.36980994 0.00019276
//   102x102    128 0.06614400 0.00455123 14.53320754 0.00020468
//   512x768     64 0.07387744 0.00931514  7.93090318 0.00026464
//   512x768    128 0.06938528 0.00944467  7.34649965 0.00026655
// 1024x1024     64 0.06564512 0.00857094  7.65903032 0.00017750
// 1024x1024    128 0.06754022 0.01263802  5.34421095 0.00016940
// 2048x1536     64 0.06559705 0.02026106  3.23759288 0.00017083
// 2048x1536    128 0.06968800 0.03764071  1.85140006 0.00024462
// 4096x4096     64 0.07459673 0.13946861  0.53486395 0.00021839
// 4096x4096    128 0.06704922 0.17351225  0.38642353 0.00020444
// 先对NF4_LUT_HALF进行处理，转化为__half
__global__ void nf4_dequant_v3(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2,
    const __half *__restrict__ code2,
    float offset,
    int64_t total_half_elements,
    int blocksize,
    int group_size,
    __half2 *__restrict__ output)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_half_elements) return;
    uint8_t val = packed[tid];
    int64_t idx = tid << 1;
    int64_t block_idx = idx / blocksize;
    int64_t group_idx = block_idx / group_size;
    __half scale = __hadd(__hmul(code2[absmax_q[block_idx]], absmax2[group_idx]),__float2half(offset) );
    __half v1 = __hmul(NF4_LUT_HALF[val >> 4], scale);
    __half v2 = __hmul(NF4_LUT_HALF[val & 0x0F], scale);

    output[tid] = __halves2half2(v1, v2);
}
// 利用共享内存，由于上一个对于packed是合并访存，但是对于absmax_q与absmax2却不是，可以采用共享内存,
__global__ void nf4_dequant_v4(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2,
    const __half *__restrict__ code2,
    float offset,
    int64_t total_half_elements,
    int blocksize,
    int group_size,
    __half2 *__restrict__ output)
{
    extern __shared__ __half shared_scale[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int64_t cta_half_start = (int64_t)blockIdx.x * blockDim.x * 2;
    int64_t cta_half_end   = min(cta_half_start + blockDim.x * 2,total_half_elements * 2);

    int first_block = cta_half_start / blocksize;
    int last_block  = (cta_half_end + blocksize - 1) / blocksize;

    int num_scale = last_block - first_block;

    for (int i = tid; i < num_scale; i += blockDim.x)
    {
        int block_idx = first_block + i;
        int group_idx = block_idx / group_size;

        __half s1 = code2[absmax_q[block_idx]];
        __half s2 = absmax2[group_idx];

        shared_scale[i] = __hadd(__hmul(s1, s2), __float2half(offset));
    }

    __syncthreads();

    if (idx >= total_half_elements) return;

    uint8_t val = packed[idx];

    int64_t half_idx = ((int64_t)idx) << 1;

    int block_idx   = half_idx / blocksize;
    int local_block = block_idx - first_block;

    __half scale = shared_scale[local_block];

    __half v1 = __hmul(NF4_LUT_HALF[val >> 4], scale);
    __half v2 = __hmul(NF4_LUT_HALF[val & 0xF], scale);

    output[idx] = __halves2half2(v1, v2);
}
// ======== NF4 Dequant Benchmark ========

//     Shape  Block   bnb (ms)  cuda (ms)     Speedup        MAE
//   256x256     64 0.07390688 0.00457235 16.16386585 0.00022089
//   256x256    128 0.07219904 0.00449523 16.06124925 0.00020719
//   512x512     64 0.07279469 0.00450157 16.17096264 0.00019741
//   512x512    128 0.07222566 0.00450477 16.03315951 0.00018454
// 1024x1024     64 0.08602036 0.00752256 11.43498408 0.00028038
// 1024x1024    128 0.07106272 0.00738086  9.62796766 0.00022340
// 2048x2048     64 0.07193088 0.02113619  3.40320889 0.00019407
// 2048x2048    128 0.07131866 0.02050112  3.47876888 0.00027776
// 4096x4096     64 0.07169056 0.07539520  0.95086367 0.00028443
// 4096x4096    128 0.07180358 0.07302701  0.98324693 0.00025392
// 8192x8192     64 0.10076653 0.28874661  0.34897907 0.00018609
// 8192x8192    128 0.09796436 0.27818124  0.35216017 0.00021183
// 进行向量化读取
__global__ void nf4_dequant_v5(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ absmax_q,
    const __half *__restrict__ absmax2,
    const __half *__restrict__ code2,
    float offset,
    int64_t total_half_elements,
    int blocksize,
    int group_size,
    __half *__restrict__ output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = total_half_elements >> 1;   // 2 half / byte
    int64_t byte_idx = (int64_t)tid * 4;
    if (byte_idx >= total_bytes) return;
    uint32_t pack4 = ((const uint32_t*)packed)[tid];
    uint8_t b0 =  pack4        & 0xFF;
    uint8_t b1 = (pack4 >> 8 ) & 0xFF;
    uint8_t b2 = (pack4 >> 16) & 0xFF;
    uint8_t b3 = (pack4 >> 24) & 0xFF;
    int64_t half_base = byte_idx << 1;
    int block_idx = half_base / blocksize;
    int group_idx = block_idx / group_size;
    __half scale = __hadd(__hmul(code2[absmax_q[block_idx]], absmax2[group_idx]), __float2half(offset));
    __half h[8];
    h[0]   = __hmul(NF4_LUT_HALF[(b0)>>4], scale); 
    h[1] = __hmul(NF4_LUT_HALF[(b0)&0xF], scale);
    h[2]   = __hmul(NF4_LUT_HALF[(b1)>>4], scale); 
    h[3] = __hmul(NF4_LUT_HALF[(b1)&0xF], scale);
    h[4]   = __hmul(NF4_LUT_HALF[(b2)>>4], scale); 
    h[5] = __hmul(NF4_LUT_HALF[(b2)&0xF], scale);
    h[6]   = __hmul(NF4_LUT_HALF[(b3)>>4], scale); 
    h[7] = __hmul(NF4_LUT_HALF[(b3)&0xF], scale);

    uint4 out_pack;
    reinterpret_cast<__half*>(&out_pack)[0] = h[0];
    reinterpret_cast<__half*>(&out_pack)[1] = h[1];
    reinterpret_cast<__half*>(&out_pack)[2] = h[2];
    reinterpret_cast<__half*>(&out_pack)[3] = h[3];
    reinterpret_cast<__half*>(&out_pack)[4] = h[4];
    reinterpret_cast<__half*>(&out_pack)[5] = h[5];
    reinterpret_cast<__half*>(&out_pack)[6] = h[6];
    reinterpret_cast<__half*>(&out_pack)[7] = h[7];

    ((uint4*)(output + half_base))[0] = out_pack;
}

__global__ void nf4_dequant_v6(const uint8_t *__restrict__ packed,
                               const uint8_t *__restrict__ absmax_q,
                               const __half *__restrict__ absmax2, float offset,
                               int64_t total_half_elements, int blocksize,
                               int group_size, __half *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_bytes = total_half_elements >> 1; // 2 half / byte
  int64_t byte_idx = (int64_t)tid * 4;
  if (byte_idx >= total_bytes)
    return;
  uint32_t pack4 = ((const uint32_t *)packed)[tid];
  uint8_t b0 = pack4 & 0xFF;
  uint8_t b1 = (pack4 >> 8) & 0xFF;
  uint8_t b2 = (pack4 >> 16) & 0xFF;
  uint8_t b3 = (pack4 >> 24) & 0xFF;
  int64_t half_base = byte_idx << 1;
  int block_idx = half_base / blocksize;
  int group_idx = block_idx / group_size;
  __half scale =
      __hadd(__hmul(CODE2_LUT[absmax_q[block_idx]], absmax2[group_idx]),
             __float2half(offset));
  __half h[8];
  h[0] = __hmul(NF4_LUT_HALF[(b0) >> 4], scale);
  h[1] = __hmul(NF4_LUT_HALF[(b0) & 0xF], scale);
  h[2] = __hmul(NF4_LUT_HALF[(b1) >> 4], scale);
  h[3] = __hmul(NF4_LUT_HALF[(b1) & 0xF], scale);
  h[4] = __hmul(NF4_LUT_HALF[(b2) >> 4], scale);
  h[5] = __hmul(NF4_LUT_HALF[(b2) & 0xF], scale);
  h[6] = __hmul(NF4_LUT_HALF[(b3) >> 4], scale);
  h[7] = __hmul(NF4_LUT_HALF[(b3) & 0xF], scale);

  uint4 out_pack;
  reinterpret_cast<__half *>(&out_pack)[0] = h[0];
  reinterpret_cast<__half *>(&out_pack)[1] = h[1];
  reinterpret_cast<__half *>(&out_pack)[2] = h[2];
  reinterpret_cast<__half *>(&out_pack)[3] = h[3];
  reinterpret_cast<__half *>(&out_pack)[4] = h[4];
  reinterpret_cast<__half *>(&out_pack)[5] = h[5];
  reinterpret_cast<__half *>(&out_pack)[6] = h[6];
  reinterpret_cast<__half *>(&out_pack)[7] = h[7];

  ((uint4 *)(output + half_base))[0] = out_pack;
}
__global__ void nf4_dequant_v8_uint4_vectorized(
    const uint4 *__restrict__ packed_uint4, // 更改类型以强制向量化加载
    const uint8_t *__restrict__ absmax_q, const __half *__restrict__ absmax2,
    const __half *__restrict__ code2, float offset, int64_t total_half_elements,
    int blocksize, int group_size, __half *__restrict__ output) {
  // 每个线程现在处理 1 个 uint4 = 16 字节 = 32 个 NF4 元素
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 基础索引计算
  int64_t half_base = (int64_t)tid * 32;
  if (half_base >= total_half_elements)
    return;

  // 1. Vectorized Load: 一次读取 128-bit (16字节)
  uint4 load_val = packed_uint4[tid];

  // 2. 预计算 Scale (假设 32 个元素都在同一个 block/group 内以简化逻辑)
  // 如果 blocksize 很小（如 16），这里需要为每组单独计算 scale
  int block_idx = (int)(half_base / blocksize);
  int group_idx = block_idx / group_size;
  __half s_raw = __hadd(__hmul(code2[absmax_q[block_idx]], absmax2[group_idx]),
                        __float2half(offset));
  __half2 scale2 = __half2half2(s_raw);

  // 3. 处理 32 个元素 (分成 4 组 uint32 进行解包)
  uint32_t chunks[4] = {load_val.x, load_val.y, load_val.z, load_val.w};

// 我们需要写回 4 个 uint4 (每个 uint4 存 8 个 half)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t pack4 = chunks[i];

    // 这里的解包逻辑使用了 half2 优化
    auto dequant_byte = [&](uint8_t byte) {
      __half2 h2;
      h2.x = NF4_LUT_HALF[byte >> 4];  // 高4位
      h2.y = NF4_LUT_HALF[byte & 0xF]; // 低4位
      return __hmul2(h2, scale2);
    };

    __half2 res0 = dequant_byte((uint8_t)(pack4 & 0xFF));
    __half2 res1 = dequant_byte((uint8_t)((pack4 >> 8) & 0xFF));
    __half2 res2 = dequant_byte((uint8_t)((pack4 >> 16) & 0xFF));
    __half2 res3 = dequant_byte((uint8_t)((pack4 >> 24) & 0xFF));

    // 4. Vectorized Store: 写回 8 个 half
    uint4 out_val;
    ((__half2 *)&out_val)[0] = res0;
    ((__half2 *)&out_val)[1] = res1;
    ((__half2 *)&out_val)[2] = res2;
    ((__half2 *)&out_val)[3] = res3;

    // 计算写回位置：每个 i 偏移 8 个 half
    ((uint4 *)(output + half_base))[i] = out_val;
  }
}
extern "C" {
void nf4_dequant_cuda_double(const uint8_t *packed, const uint8_t *absmax_q,
                             const __half *absmax2, const __half *code2,
                             float offset, int64_t total, int blocksize,
                             int group_size, __half *output) {
//   int64_t num_units = (total + 1) / 2;
//   int threads = 256;
//   int blocks = (num_units + threads - 1) / threads;
//   // 获得不同scale值的数目
//   int sum_scale =(threads * 2 + blocksize - 1) / blocksize;
//   size_t smem_size = sum_scale * sizeof(__half);
//   nf4_dequant_v2<<<blocks, threads,smem_size>>>(
//       packed, absmax_q, absmax2, code2, offset, num_units, blocksize,
//       group_size, (__half2 *)output);
  // 采用向量化
  int threads = 256;
  int64_t total_bytes = total >> 1;
  int blocks = (total_bytes/4 + threads - 1) / threads;
  nf4_dequant_v5<<<blocks, threads>>>(
    packed, absmax_q, absmax2, code2, offset, total, blocksize,
    group_size, output);


}
void init_nf4_lut()
{
    float lut_f[16] = {
        -1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
        -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
         0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
         0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f
    };

    __half lut_h[16];

    for(int i=0;i<16;i++)
        lut_h[i] = __float2half(lut_f[i]);
    
    cudaMemcpyToSymbol(NF4_LUT_HALF, lut_h, sizeof(lut_h));
}
}