#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <algorithm>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// 配置结构体
struct AppConfig {
    int blocksize = 64;
    std::string compute_type = "fp16";
    std::string target_gpu = "T4";
};

// 配置解析函数
AppConfig load_config(const std::string& filename) {
    AppConfig config;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << "[Config] No config.txt found, using defaults.\n";
        return config;
    }

    std::string line;
    while (std::getline(f, line)) {
        std::replace(line.begin(), line.end(), '=', ' ');
        std::stringstream ss(line);
        std::string key, value;
        if (ss >> key >> value) {
            if (key == "blocksize") config.blocksize = std::stoi(value);
            else if (key == "compute_type") config.compute_type = value;
            else if (key == "target_gpu") config.target_gpu = value;
        }
    }
    std::cout << "[Config] Loaded: compute_type=" << config.compute_type 
              << ", target_gpu=" << config.target_gpu << "\n";
    return config;
}

// NF4 查找表：放入 __constant__ 内存
__constant__ float d_NF4_TABLE[16] = {
    -1.0f, -0.69487101f, -0.51209301f, -0.37391701f,
    -0.25611401f, -0.14725500f, -0.04162400f, 0.06282201f,
    0.16859101f, 0.28551400f, 0.40619302f, 0.53675699f,
    0.68502200f, 0.87091398f, 1.0f, 0.0f
};

// V3 Kernel: Shared Memory + 向量化访存
__global__ void dequantize_nf4_kernel_v3(
    const uint8_t* packed_w,
    const uint8_t* absmax_q,
    const half* code2,
    const half* absmax2,
    half2* output,
    int64_t total_elements,
    int block_size,
    int group_size
) {
    // 1. 将 code2 放入共享内存
    __shared__ half s_code2[256];
    int tid = threadIdx.x;
    if (tid < 256) {
        s_code2[tid] = code2[tid];
    }
    __syncthreads();

    int64_t byte_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= (total_elements + 1) / 2) return;

    // 解包 1 个 byte 为两个 4-bit 索引
    uint8_t byte = packed_w[byte_idx];
    
    int64_t element_idx = byte_idx * 2; 
    int32_t b_idx = element_idx / block_size;
    int32_t g_idx = b_idx / group_size;

    float s1 = __half2float(s_code2[absmax_q[b_idx]]);
    float s2 = __half2float(absmax2[g_idx]);
    float scale = s1 * s2;

    // 向量化写入
    half res0 = __float2half(d_NF4_TABLE[byte & 0x0F] * scale);
    half res1 = __float2half(d_NF4_TABLE[byte >> 4] * scale);

    // 边界保护与写入
    if (byte_idx * 2 + 1 < total_elements) {
        output[byte_idx] = make_half2(res0, res1);
    } else if (byte_idx * 2 < total_elements) {
        // 边界情况：处理奇数长度矩阵的最后一个元素，退化为 16-bit 标量写入
        reinterpret_cast<half*>(output)[byte_idx * 2] = res0;
    }

}

int main() {
    AppConfig cfg = load_config("config.txt");

    std::ifstream ifs("input.bin", std::ios::binary);
    if (!ifs) { std::cerr << "Cannot open input.bin\n"; return 1; }
    

    int64_t num_rows, num_cols;
    int32_t blocksize;
    ifs.read((char*)&num_rows, 8);
    ifs.read((char*)&num_cols, 8);
    ifs.read((char*)&blocksize, 4);

    int64_t total_elements = num_rows * num_cols;
    int32_t num_blocks = (total_elements + blocksize - 1) / blocksize;
    int32_t group_size = 256;
    int32_t num_groups = (num_blocks + group_size - 1) / group_size;

    std::vector<uint8_t> h_packed_w((total_elements + 1) / 2);
    std::vector<uint8_t> h_absmax_q(num_blocks);
    std::vector<half> h_code2(256);
    std::vector<half> h_absmax2(num_groups);
    float offset;

    ifs.read((char*)h_packed_w.data(), h_packed_w.size());
    ifs.read((char*)h_absmax_q.data(), h_absmax_q.size());
    ifs.read((char*)h_code2.data(), 256 * 2);
    ifs.read((char*)h_absmax2.data(), num_groups * 2);
    ifs.read((char*)&offset, 4);

    if (cfg.blocksize != blocksize) {
        printf("[Warning] Binary header blocksize (%d) differs from config.txt (%d). "
               "Using Binary Header.\n", blocksize, cfg.blocksize);
    }

    int threads_per_block = 256; // 默认值
    if (cfg.target_gpu == "T4") {
        // T4 (Turing) 架构 SM 较小，128 是更合适的选择
        threads_per_block = 128;
    } else {
        //  4060 或 A100 等现代显卡，256 是甜点值
        threads_per_block = 256;
    }

    uint8_t *d_packed_w, *d_absmax_q;
    half *d_code2, *d_absmax2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_packed_w, h_packed_w.size()));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, h_absmax_q.size()));
    CHECK_CUDA(cudaMalloc(&d_code2, 256 * 2));
    CHECK_CUDA(cudaMalloc(&d_absmax2, num_groups * 2));
    CHECK_CUDA(cudaMalloc(&d_output, total_elements * 2));

    CHECK_CUDA(cudaMemcpy(d_packed_w, h_packed_w.data(), h_packed_w.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), h_absmax_q.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2, h_code2.data(), 256 * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, h_absmax2.data(), num_groups * 2, cudaMemcpyHostToDevice));

    // 计时
    int64_t num_bytes = (total_elements + 1) / 2;
    int blocks = (num_bytes + threads_per_block - 1) / threads_per_block;

    if (cfg.compute_type == "bf16") {
        // 打印 Dispatch 日志
        std::cout << "[Dispatch] Launching Kernel with BF16 precision path (simulated)...\n";
    } else {
        std::cout << "[Dispatch] Launching Kernel with FP16 precision path...\n";
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热 GPU (避免把 cuda context 初始化算进时间)
    dequantize_nf4_kernel_v3<<<blocks, threads_per_block>>>(
        d_packed_w, d_absmax_q, d_code2, d_absmax2, 
        (half2*)d_output, total_elements, blocksize, group_size
    );
    cudaDeviceSynchronize();

    // 正式计时 (跑 10 次取平均)
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i) { 
        dequantize_nf4_kernel_v3<<<blocks, threads_per_block>>>(
            d_packed_w, d_absmax_q, d_code2, d_absmax2, 
            (half2*)d_output, total_elements, blocksize, group_size
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f; // 取 10 次的平均值

// 验证结果
    std::vector<half> h_output(total_elements);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total_elements * 2, cudaMemcpyDeviceToHost));

    // 读取 Ground Truth
    std::vector<half> h_gt(total_elements);
    std::ifstream gfs("gt_output.bin", std::ios::binary);
    gfs.read((char*)h_gt.data(), total_elements * 2);

    float max_err = 0;
    double total_abs_err = 0;
    for(int i=0; i<total_elements; ++i) {
        float out_f = __half2float(h_output[i]);
        float gt_f = __half2float(h_gt[i]);
        float diff = std::abs(out_f - gt_f);
        
        max_err = std::max(max_err, diff);
        total_abs_err += diff;
    }
    float mae = static_cast<float>(total_abs_err / total_elements);

    std::cout << "--- Validation Results ---" << std::endl;
    std::cout << "CUDA V3 Max Error: " << max_err << std::endl;
    std::cout << "CUDA V3 MAE:       " << mae << std::endl;
    std::cout << "Time:              " << ms << " ms" << std::endl;
    
    // 计算有效内存带宽
    // 读取: W(4bit) + absmax_q(8bit) + absmax2(16bit)
    // 写入: Output(16bit)
    // 忽略 code2 (已经放入 Shared Memory) 的重复读取开销
    double bytes_read = (total_elements * 0.5) + num_blocks + (num_groups * 2.0);
    double bytes_write = total_elements * 2.0;
    double total_bytes = bytes_read + bytes_write;
    
    std::cout << "Bandwidth: " << total_bytes / (ms * 1e6) << " GB/s" << std::endl;

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packed_w); cudaFree(d_absmax_q); cudaFree(d_code2); cudaFree(d_absmax2); cudaFree(d_output);
    return 0;
}