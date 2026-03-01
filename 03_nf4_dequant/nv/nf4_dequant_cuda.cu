#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> 
#include <sys/time.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 将表放入到常量内存加快速度
__constant__ __half NF4_LUT_HALF[16];
__constant__ __half CODE2_LUT[256];

void ensure_directory_exists(const char *path) {
  struct stat st = {0};
  if (stat(path, &st) == -1) {
#ifdef _WIN32
    mkdir(path);
#else
    mkdir(path, 0755);
#endif
  }
}

// ============================================================
// 初始化 LUT (将 float 转换为 half)
// ============================================================
void init_nf4_lut() {
  float lut_f[16] = {-1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
                     -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
                     0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
                     0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f};

  __half lut_h[16];
  for (int i = 0; i < 16; i++) {
    lut_h[i] = __float2half(lut_f[i]);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(NF4_LUT_HALF, lut_h, sizeof(lut_h)));
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
// 读取权重文件
int read_weight_file(const char *filename, int64_t *rows, int64_t *cols,
                     int *blocksize, uint8_t **packed, uint8_t **absmax_q,
                     __half **absmax2, __half **code2, float *offset) {

  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, " 无法打开文件: %s\n", filename);
    return -1;
  }

  // 读取 header
  fread(rows, sizeof(int64_t), 1, fp);
  fread(cols, sizeof(int64_t), 1, fp);
  fread(blocksize, sizeof(int32_t), 1, fp);

  int64_t total_elements = (*rows) * (*cols);
  int64_t num_packed = (total_elements + 1) / 2;
  int64_t num_blocks = (total_elements + *blocksize - 1) / *blocksize;
  int64_t num_groups = (num_blocks + 255) / 256;

  printf("\n 文件信息:\n");
  printf("  矩阵: %ld x %ld\n", *rows, *cols);
  printf("  总元素数: %ld\n", total_elements);
  printf("  blocksize: %d\n", *blocksize);
  printf("  打包数据大小: %ld bytes\n", num_packed);
  printf("  量化块数: %ld\n", num_blocks);
  printf("  分组数: %ld\n", num_groups);

  // 分配内存
  *packed = (uint8_t *)malloc(num_packed);
  *absmax_q = (uint8_t *)malloc(num_blocks);
  *absmax2 = (__half *)malloc(num_groups * sizeof(__half));
  *code2 = (__half *)malloc(256 * sizeof(__half));

  if (!*packed || !*absmax_q || !*absmax2 || !*code2) {
    fprintf(stderr, " 主机内存分配失败\n");
    fclose(fp);
    return -1;
  }

  // 读取数据
  fread(*packed, 1, num_packed, fp);
  fread(*absmax_q, 1, num_blocks, fp);
  fread(*absmax2, sizeof(__half), num_groups, fp);
  fread(*code2, sizeof(__half), 256, fp);
  fread(offset, sizeof(float), 1, fp);

  fclose(fp);
  printf(" 文件读取成功\n");
  return 0;
}

// 保存解量化后的权重（自动保存到cuda_results目录）
void save_dequantized_weight(const char *filename, __half *weight,
                             int64_t total_elements) {
  ensure_directory_exists("../cuda_results");

  // 构建完整路径
  char full_path[512];
  snprintf(full_path, sizeof(full_path), "../cuda_results/%s", filename);

  FILE *fp = fopen(full_path, "wb");
  if (!fp) {
    fprintf(stderr, " 无法创建输出文件: %s\n", full_path);
    return;
  }

  fwrite(weight, sizeof(__half), total_elements, fp);
  fclose(fp);

  printf(" 已保存解量化结果: %s (%.2f MB)\n", full_path,
         (total_elements * sizeof(__half)) / (1024.0 * 1024.0));
}


// 计时器 (毫秒)
double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}


// 计算有效内存带宽
double calculate_bandwidth(int64_t total_elements, double time_ms) {
  // 输入数据大小
  int64_t input_bytes = (total_elements + 1) / 2; // packed
  input_bytes += (total_elements + 64 - 1) / 64;  // absmax_q (近似)
  input_bytes +=
      ((total_elements + 64 - 1) / 64 + 255) / 256 * sizeof(__half); // absmax2
  input_bytes += 256 * sizeof(__half);                               // code2

  // 输出数据大小
  int64_t output_bytes = total_elements * sizeof(__half);

  int64_t total_bytes = input_bytes + output_bytes;

  return (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char **argv) {

  if (argc != 2) {
    printf("\n使用方法: %s <权重文件.bin>\n", argv[0]);
    printf("  权重文件格式: 由 Python 脚本生成的 .bin 文件\n");
    printf("  示例: %s weight_data/weight_1024x1024_bs64.bin\n\n", argv[0]);
    return -1;
  }

  const char *input_file = argv[1];

  // 确保输出目录存在
  ensure_directory_exists("../cuda_results");

  // 初始化 LUT
  printf("\n 初始化 NF4 LUT...\n");
  init_nf4_lut();

  // 读取权重文件
  printf("\n 读取权重文件: %s\n", input_file);
  int64_t rows, cols;
  int blocksize;
  uint8_t *h_packed, *h_absmax_q;
  __half *h_absmax2, *h_code2;
  float offset;

  if (read_weight_file(input_file, &rows, &cols, &blocksize, &h_packed,
                       &h_absmax_q, &h_absmax2, &h_code2, &offset) != 0) {
    return -1;
  }

  int64_t total_elements = rows * cols;
  int64_t num_units = (total_elements + 1) / 2; // 每个 uint8 包含两个 half

  // 计算 GPU 内存大小
  int64_t num_blocks = (total_elements + blocksize - 1) / blocksize;
  int64_t num_groups = (num_blocks + 255) / 256;

  printf("\n 计算参数:\n");
  printf("  total_elements: %ld\n", total_elements);
  printf("  num_units: %ld\n", num_units);
  printf("  num_blocks: %ld\n", num_blocks);
  printf("  num_groups: %ld\n", num_groups);

  // 分配 GPU 内存
  printf("\n 分配 GPU 内存...\n");
  uint8_t *d_packed, *d_absmax_q;
  __half *d_absmax2, *d_code2, *d_output;

  CUDA_CHECK(cudaMalloc(&d_packed, num_units));
  CUDA_CHECK(cudaMalloc(&d_absmax_q, num_blocks));
  CUDA_CHECK(cudaMalloc(&d_absmax2, num_groups * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_code2, 256 * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_output, total_elements * sizeof(__half)));

  // 拷贝数据到 GPU
  printf(" 拷贝数据到 GPU...\n");
  CUDA_CHECK(cudaMemcpy(d_packed, h_packed, num_units, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_absmax_q, h_absmax_q, num_blocks, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_absmax2, h_absmax2, num_groups * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_code2, h_code2, 256 * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(CODE2_LUT, h_code2, 256 * sizeof(__half)));

  // 分配主机输出内存
  __half *h_output = (__half *)malloc(total_elements * sizeof(__half));
  if (!h_output) {
    fprintf(stderr, " 主机输出内存分配失败\n");
    return -1;
  }

  // 配置内核启动参数
  int threads = 256;
  int64_t total_bytes = total_elements >> 1;
  int blocks = (total_bytes / 4 + threads - 1) / threads;
  printf("\n 内核配置:\n");
  printf("  blocks: %d\n", blocks);
  printf("  threads per block: %d\n", threads);
  printf("  总线程数: %d\n", blocks * threads);

  // 预热 (5次)
  printf("\n 预热 (5次)...\n");
  for (int i = 0; i < 5; i++) {
    nf4_dequant_v6<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, offset,
                                        total_elements, blocksize, 256,
                                        d_output);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 正式测试 (100次)
  printf("  性能测试 (100次迭代)...\n");
  double start_time = get_time_ms();

  for (int i = 0; i < 100; i++) {
    nf4_dequant_v6<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, offset,
                                        total_elements, blocksize, 256,
                                        d_output);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  double end_time = get_time_ms();

  double total_time = end_time - start_time;
  double avg_time_ms = total_time / 100.0;

  // 计算带宽
  double bandwidth = calculate_bandwidth(total_elements, avg_time_ms);

  // 拷贝结果回主机
  printf("🔄 拷贝结果回主机...\n");
  CUDA_CHECK(cudaMemcpy(h_output, d_output, total_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));

  // 生成输出文件名
  char output_file[256];
  snprintf(output_file, sizeof(output_file), "dequant_%ldx%ld_bs%d.fp16", rows,
           cols, blocksize);

  // 保存解量化结果（自动保存到cuda_results目录）
  printf("\n 保存解量化结果...\n");
  save_dequantized_weight(output_file, h_output, total_elements);

  // 生成性能日志文件名
  char log_file[256];
  snprintf(log_file, sizeof(log_file), "perf_%ldx%ld_bs%d.log", rows, cols,
           blocksize);

  // 输出性能结果
  printf("输入文件: %s\n", input_file);
  printf("矩阵大小: %ld x %ld\n", rows, cols);
  printf("总元素数: %ld\n", total_elements);
  printf("数据大小: %.2f MB\n",
         total_elements * sizeof(__half) / (1024.0 * 1024.0));
  printf("\n");
  printf("核函数执行时间: %.4f ms\n", avg_time_ms);
  printf("有效内存带宽: %.2f GB/s\n", bandwidth);
  printf("\n");
  printf("输出文件: cuda_results/%s\n", output_file);
  printf("日志文件: cuda_results/%s\n", log_file);


  // 保存性能日志（也保存到cuda_results目录）
  char log_path[512];
  snprintf(log_path, sizeof(log_path), "../cuda_results/%s", log_file);

  FILE *log_fp = fopen(log_path, "w");
  if (log_fp) {
    fprintf(log_fp, "input_file=%s\n", input_file);
    fprintf(log_fp, "rows=%ld\n", rows);
    fprintf(log_fp, "cols=%ld\n", cols);
    fprintf(log_fp, "blocksize=%d\n", blocksize);
    fprintf(log_fp, "total_elements=%ld\n", total_elements);
    fprintf(log_fp, "kernel_time_ms=%.4f\n", avg_time_ms);
    fprintf(log_fp, "bandwidth_gbps=%.2f\n", bandwidth);
    fprintf(log_fp, "output_file=cuda_results/%s\n", output_file);
    fclose(log_fp);
    printf(" 性能日志已保存到: %s\n", log_path);
  }

  // 清理
  free(h_packed);
  free(h_absmax_q);
  free(h_absmax2);
  free(h_code2);
  free(h_output);

  cudaFree(d_packed);
  cudaFree(d_absmax_q);
  cudaFree(d_absmax2);
  cudaFree(d_code2);
  cudaFree(d_output);

  printf("\n 测试完成!\n\n");
  return 0;
}