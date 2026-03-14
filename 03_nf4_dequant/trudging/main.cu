#include "src/weights_loader.h"
#include "src/dequantize.c.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 辅助宏：用于检查 CUDA 错误
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main(int argc, char** argv) {
    std::cout << "Starting NF4 Dequantization Kernel Test..." << std::endl;

    // 1. 读取量化权重文件
    std::string weights_file = "test_weights.bin";
    std::string gt_file = "ground_truth.bin";
    
    std::cout << "Loading weights from " << weights_file << "..." << std::endl;
    QuantizedWeights gt_weights;
    try {
        gt_weights = load_weights(weights_file);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load weights: " << e.what() << std::endl;
        return -1;
    }

    int64_t num_rows = gt_weights.num_rows;
    int64_t num_cols = gt_weights.num_cols;
    int blocksize = gt_weights.block_size;

    int64_t total_elements = num_rows * num_cols;
    int64_t num_blocks = gt_weights.num_blocks;
    int64_t num_groups = gt_weights.num_groups;
    int64_t packed_size = gt_weights.packed_size;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Matrix: " << num_rows << " x " << num_cols << " (" << total_elements << " elements)" << std::endl;
    std::cout << "  Blocksize: " << blocksize << std::endl;
    std::cout << "  Num Blocks: " << num_blocks << std::endl;
    std::cout << "  Num Groups: " << num_groups << std::endl;
    std::cout << "  Packed Size: " << packed_size << " bytes" << std::endl;

    // 读取 Ground Truth
    std::cout << "Loading ground truth from " << gt_file << "..." << std::endl;
    std::vector<uint16_t> h_ground_truth(total_elements); // store as fp16 bits
    std::ifstream f_gt(gt_file, std::ios::binary);
    if (!f_gt.is_open()) {
        std::cerr << "Failed to open " << gt_file << std::endl;
        return -1;
    }
    f_gt.read(reinterpret_cast<char*>(h_ground_truth.data()), total_elements * sizeof(uint16_t));
    f_gt.close();

    // 2. 显存分配与数据拷贝 (H2D)
    uint8_t *d_packed_weights, *d_absmax_q;
    uint16_t *d_absmax2, *d_code2;
    __nv_bfloat16 *d_output;

    CHECK_CUDA(cudaMalloc(&d_packed_weights, packed_size * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, num_blocks * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_absmax2, num_groups * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_code2, 256 * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_output, total_elements * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_packed_weights, gt_weights.packed_weights.get(), packed_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, gt_weights.absmax_q.get(), num_blocks * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, gt_weights.absmax2.get(), num_groups * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2, gt_weights.code2.get(), 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // 3. 性能测速 (CUDA Events)
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "\nStarting Warmup..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        launch_dequantize_nf4(
            d_packed_weights, d_absmax_q, d_absmax2, d_code2, 
            gt_weights.offset, d_output, total_elements, blocksize, nullptr
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Starting Profiling..." << std::endl;
    int num_runs = 100;
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        launch_dequantize_nf4(
            d_packed_weights, d_absmax_q, d_absmax2, d_code2, 
            gt_weights.offset, d_output, total_elements, blocksize, nullptr
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_time_ms = total_ms / num_runs;

    // 4. 有效内存带宽计算
    double total_bytes = packed_size 
                       + num_blocks 
                       + (num_groups * 2.0) 
                       + (256.0 * 2.0) 
                       + (total_elements * 2.0);

    double bandwidth_GBs = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << "Average Execution Time: " << std::fixed << std::setprecision(4) << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective Bandwidth:    " << std::setprecision(2) << bandwidth_GBs << " GB/s" << std::endl;

    // 5. 精度验证 (MAE)
    std::cout << "\n--- Accuracy Verification ---" << std::endl;
    std::vector<__nv_bfloat16> h_output_test(total_elements);
    CHECK_CUDA(cudaMemcpy(h_output_test.data(), d_output, total_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    
    double total_error = 0.0;
    float max_error = 0.0f;
    
    for (int64_t i = 0; i < total_elements; ++i) {
        // Ground truth was saved as IEEE fp16, convert to float
        __half gt_half;
        memcpy(&gt_half, &h_ground_truth[i], sizeof(uint16_t));
        float gt_val = __half2float(gt_half);
        
        // Output was generated as bf16, convert to float
        float out_val = __bfloat162float(h_output_test[i]);
        
        float err = std::abs(gt_val - out_val);
        total_error += err;
        if (err > max_error) {
            max_error = err;
        }
    }
    
    double mae = total_error / total_elements;
    std::cout << "Calculated elements: " << total_elements << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << std::scientific << mae << std::endl;
    std::cout << "Max Absolute Error (MaxAE): " << max_error << std::endl;
    
    if (mae < 1e-2) {
         std::cout << "=> Accuracy Check PASSED!" << std::endl;
    } else {
         std::cout << "=> Accuracy Check WARNING (MAE might be high)" << std::endl;
    }

    // 6. 资源释放
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_packed_weights));
    CHECK_CUDA(cudaFree(d_absmax_q));
    CHECK_CUDA(cudaFree(d_absmax2));
    CHECK_CUDA(cudaFree(d_code2));
    CHECK_CUDA(cudaFree(d_output));

    std::cout << "\nDone!" << std::endl;
    return 0;
}
