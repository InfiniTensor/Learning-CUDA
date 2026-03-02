#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "GPU error at %s:%d - %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__constant__ __half NF4_LUT_HALF[16];
__constant__ __half CODE2_LUT[256];

bool read_exact(FILE *fp, void *dst, size_t elem_size, size_t count) {
  return fread(dst, elem_size, count, fp) == count;
}

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

void init_nf4_lut() {
  float lut_f[16] = {-1.00000000f, -0.69619280f, -0.52507305f, -0.39491710f,
                     -0.28444138f, -0.18477343f, -0.09105003f, 0.00000000f,
                     0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
                     0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f};

  __half lut_h[16];
  for (int i = 0; i < 16; ++i) {
    lut_h[i] = __float2half(lut_f[i]);
  }
  GPU_CHECK(cudaMemcpyToSymbol(NF4_LUT_HALF, lut_h, sizeof(lut_h)));
}

__global__ void nf4_dequant_v6(const uint8_t *__restrict__ packed,
                               const uint8_t *__restrict__ absmax_q,
                               const __half *__restrict__ absmax2, float offset,
                               int64_t total_elements, int blocksize,
                               int group_size, __half *__restrict__ output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_bytes = (total_elements + 1) >> 1;
  int64_t byte_base = static_cast<int64_t>(tid) * 4;
  if (byte_base >= total_bytes)
    return;

  uint8_t bytes[4] = {0, 0, 0, 0};
  int64_t remain = total_bytes - byte_base;
  int valid = static_cast<int>(remain < 4 ? remain : 4);

  if (valid == 4) {
    uint32_t pack4 = reinterpret_cast<const uint32_t *>(packed)[tid];
    bytes[0] = pack4 & 0xFF;
    bytes[1] = (pack4 >> 8) & 0xFF;
    bytes[2] = (pack4 >> 16) & 0xFF;
    bytes[3] = (pack4 >> 24) & 0xFF;
  } else {
    for (int i = 0; i < valid; ++i) {
      bytes[i] = packed[byte_base + i];
    }
  }

  __half h_offset = __float2half(offset);

  for (int i = 0; i < valid; ++i) {
    int64_t elem_base = (byte_base + i) << 1;
    if (elem_base >= total_elements)
      break;

    int block_idx = static_cast<int>(elem_base / blocksize);
    int group_idx = block_idx / group_size;
    __half scale =
        __hadd(__hmul(CODE2_LUT[absmax_q[block_idx]], absmax2[group_idx]), h_offset);

    uint8_t b = bytes[i];
    output[elem_base] = __hmul(NF4_LUT_HALF[(b >> 4) & 0xF], scale);
    if (elem_base + 1 < total_elements) {
      output[elem_base + 1] = __hmul(NF4_LUT_HALF[b & 0xF], scale);
    }
  }
}

int read_weight_file(const char *filename, int64_t *rows, int64_t *cols,
                     int *blocksize, uint8_t **packed, uint8_t **absmax_q,
                     __half **absmax2, __half **code2, float *offset) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "无法打开文件: %s\n", filename);
    return -1;
  }

  if (!read_exact(fp, rows, sizeof(int64_t), 1) ||
      !read_exact(fp, cols, sizeof(int64_t), 1) ||
      !read_exact(fp, blocksize, sizeof(int32_t), 1)) {
    fprintf(stderr, "读取文件头失败: %s\n", filename);
    fclose(fp);
    return -1;
  }

  int64_t total_elements = (*rows) * (*cols);
  int64_t num_packed = (total_elements + 1) / 2;
  int64_t num_blocks = (total_elements + *blocksize - 1) / *blocksize;
  int64_t num_groups = (num_blocks + 255) / 256;

  printf("\n文件信息:\n");
  printf("  矩阵: %ld x %ld\n", *rows, *cols);
  printf("  总元素数: %ld\n", total_elements);
  printf("  blocksize: %d\n", *blocksize);
  printf("  打包数据大小: %ld bytes\n", num_packed);
  printf("  量化块数: %ld\n", num_blocks);
  printf("  分组数: %ld\n", num_groups);

  *packed = (uint8_t *)malloc(num_packed);
  *absmax_q = (uint8_t *)malloc(num_blocks);
  *absmax2 = (__half *)malloc(num_groups * sizeof(__half));
  *code2 = (__half *)malloc(256 * sizeof(__half));

  if (!*packed || !*absmax_q || !*absmax2 || !*code2) {
    fprintf(stderr, "主机内存分配失败\n");
    fclose(fp);
    return -1;
  }

  if (!read_exact(fp, *packed, 1, num_packed) ||
      !read_exact(fp, *absmax_q, 1, num_blocks) ||
      !read_exact(fp, *absmax2, sizeof(__half), num_groups) ||
      !read_exact(fp, *code2, sizeof(__half), 256) ||
      !read_exact(fp, offset, sizeof(float), 1)) {
    fprintf(stderr, "读取量化数据失败: %s\n", filename);
    fclose(fp);
    return -1;
  }

  fclose(fp);
  printf("文件读取成功\n");
  return 0;
}

void save_dequantized_weight(const char *filename, __half *weight,
                             int64_t total_elements) {
  ensure_directory_exists("../mx_results");

  char full_path[512];
  snprintf(full_path, sizeof(full_path), "../mx_results/%s", filename);

  FILE *fp = fopen(full_path, "wb");
  if (!fp) {
    fprintf(stderr, "无法创建输出文件: %s\n", full_path);
    return;
  }

  fwrite(weight, sizeof(__half), total_elements, fp);
  fclose(fp);

  printf("已保存解量化结果: %s (%.2f MB)\n", full_path,
         (total_elements * sizeof(__half)) / (1024.0 * 1024.0));
}

double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

double calculate_bandwidth(int64_t total_elements, double time_ms, int blocksize) {
  int64_t input_bytes = (total_elements + 1) / 2;
  int64_t num_blocks = (total_elements + blocksize - 1) / blocksize;
  int64_t num_groups = (num_blocks + 255) / 256;
  input_bytes += num_blocks;
  input_bytes += num_groups * sizeof(__half);
  input_bytes += 256 * sizeof(__half);

  int64_t output_bytes = total_elements * sizeof(__half);
  int64_t total_bytes = input_bytes + output_bytes;

  return (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("\n使用方法: %s <权重文件.bin>\n", argv[0]);
    printf("示例: %s ../weight_data/weight_1024x1024_bs64.bin\n\n", argv[0]);
    return -1;
  }

  const char *input_file = argv[1];

  ensure_directory_exists("../mx_results");
  init_nf4_lut();

  int64_t rows, cols;
  int blocksize;
  uint8_t *h_packed = nullptr;
  uint8_t *h_absmax_q = nullptr;
  __half *h_absmax2 = nullptr;
  __half *h_code2 = nullptr;
  float offset = 0.f;

  if (read_weight_file(input_file, &rows, &cols, &blocksize, &h_packed,
                       &h_absmax_q, &h_absmax2, &h_code2, &offset) != 0) {
    return -1;
  }

  int64_t total_elements = rows * cols;
  int64_t num_packed = (total_elements + 1) / 2;
  int64_t num_blocks = (total_elements + blocksize - 1) / blocksize;
  int64_t num_groups = (num_blocks + 255) / 256;

  uint8_t *d_packed = nullptr;
  uint8_t *d_absmax_q = nullptr;
  __half *d_absmax2 = nullptr;
  __half *d_code2 = nullptr;
  __half *d_output = nullptr;

  GPU_CHECK(cudaMalloc(&d_packed, num_packed));
  GPU_CHECK(cudaMalloc(&d_absmax_q, num_blocks));
  GPU_CHECK(cudaMalloc(&d_absmax2, num_groups * sizeof(__half)));
  GPU_CHECK(cudaMalloc(&d_code2, 256 * sizeof(__half)));
  GPU_CHECK(cudaMalloc(&d_output, total_elements * sizeof(__half)));

  GPU_CHECK(cudaMemcpy(d_packed, h_packed, num_packed, cudaMemcpyHostToDevice));
  GPU_CHECK(
      cudaMemcpy(d_absmax_q, h_absmax_q, num_blocks, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_absmax2, h_absmax2, num_groups * sizeof(__half),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(
      cudaMemcpy(d_code2, h_code2, 256 * sizeof(__half), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpyToSymbol(CODE2_LUT, h_code2, 256 * sizeof(__half)));

  __half *h_output = (__half *)malloc(total_elements * sizeof(__half));
  if (!h_output) {
    fprintf(stderr, "主机输出内存分配失败\n");
    return -1;
  }

  int threads = 256;
  int blocks = static_cast<int>(((num_packed + 3) / 4 + threads - 1) / threads);

  for (int i = 0; i < 5; ++i) {
    nf4_dequant_v6<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, offset,
                                        total_elements, blocksize, 256,
                                        d_output);
  }
  GPU_CHECK(cudaDeviceSynchronize());

  double start_time = get_time_ms();
  for (int i = 0; i < 100; ++i) {
    nf4_dequant_v6<<<blocks, threads>>>(d_packed, d_absmax_q, d_absmax2, offset,
                                        total_elements, blocksize, 256,
                                        d_output);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  double end_time = get_time_ms();

  double avg_time_ms = (end_time - start_time) / 100.0;
  double bandwidth = calculate_bandwidth(total_elements, avg_time_ms, blocksize);

  GPU_CHECK(cudaMemcpy(h_output, d_output, total_elements * sizeof(__half),
                       cudaMemcpyDeviceToHost));

  char output_file[256];
  snprintf(output_file, sizeof(output_file), "dequant_%ldx%ld_bs%d.fp16", rows,
           cols, blocksize);
  save_dequantized_weight(output_file, h_output, total_elements);

  char log_file[256];
  snprintf(log_file, sizeof(log_file), "perf_%ldx%ld_bs%d.log", rows, cols,
           blocksize);

  char log_path[512];
  snprintf(log_path, sizeof(log_path), "../mx_results/%s", log_file);
  FILE *log_fp = fopen(log_path, "w");
  if (log_fp) {
    fprintf(log_fp, "input_file=%s\n", input_file);
    fprintf(log_fp, "rows=%ld\n", rows);
    fprintf(log_fp, "cols=%ld\n", cols);
    fprintf(log_fp, "blocksize=%d\n", blocksize);
    fprintf(log_fp, "total_elements=%ld\n", total_elements);
    fprintf(log_fp, "kernel_time_ms=%.4f\n", avg_time_ms);
    fprintf(log_fp, "bandwidth_gbps=%.2f\n", bandwidth);
    fprintf(log_fp, "output_file=mx_results/%s\n", output_file);
    fclose(log_fp);
  }

  printf("\n输入文件: %s\n", input_file);
  printf("矩阵大小: %ld x %ld\n", rows, cols);
  printf("核函数执行时间: %.4f ms\n", avg_time_ms);
  printf("有效内存带宽: %.2f GB/s\n", bandwidth);
  printf("输出文件: mx_results/%s\n", output_file);

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

  return 0;
}
