#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <cmath>

__constant__ float _E4M3_TABLE[128];

__device__ int _float_to_e4m3(float val) {
    int idx = 0;
    if (val == 0) return idx;
    int sign = (val < 0) ? 0x80 : 0;
    float a =fabsf(val);
    int best_idx = 0;
    float best_d = fabsf(a - _E4M3_TABLE[0]);
    for (idx = 1; idx < 128; idx++) {
        float d = fabsf(a - _E4M3_TABLE[idx]);
        if (d < best_d) {
            best_d = d; 
            best_idx = idx;
        }
    }
    return sign | best_idx;
}

__global__ void quant_mxfp8_kernel(
    const float* data,
    float* scales,
    uint8_t* packed,
    const int rows, const int cols, 
    const int block_size,
    const int scale_mode,
    const float gmax
) {
    int r = blockIdx.y * blockDim.x + threadIdx.x;
    int c = blockIdx.x;
    int stride = r*cols + c*block_size;
    int num = (cols + block_size - 1) / block_size; // 这个是缩放因子的个数

    if (r >= rows) return;

    float global_scale = 0.0f;
    if (scale_mode) {
        global_scale = (gmax > 0) ? (gmax / 448.0) : 1.0;
    } else {
        float bmax = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float temp = fabsf(data[i+stride]);
            if (bmax < temp) bmax = temp;
        }
        global_scale = (bmax > 0) ? (bmax / 448.0) : 1.0;
    }
    scales[r * num + c] = global_scale;

    // 量化
    for (int i = 0; i < block_size; i++) {
        packed[r*cols + c*block_size + i] = _float_to_e4m3(data[i+stride] / scales[r*num + c]);
    }
}

__global__ void dequant_mxfp8_kernel(
    const uint8_t* packed,
    float* out,
    const float* scales,
    const int rows, const int cols, const int block_size
) {
    int r = blockIdx.y * blockDim.x + threadIdx.x;
    int c = blockIdx.x;
    
    if (r >= rows) return;
    
    int stride = r*cols + c*block_size;
    int num = (cols + block_size - 1) / block_size;
    float s = scales[r*num + c];

    for (int i = 0; i < block_size; i++) {
        int code = packed[r*cols + c*block_size + i];
        float val = _E4M3_TABLE[code & 0x7F];
        out[i + stride] = s * ((code & 0x80) ? -val : val);
    }
}

std::tuple<torch::Tensor, torch::Tensor> quant_mxfp8(
    torch::Tensor data, 
    int64_t rows, int64_t cols, int64_t block_size,
    int64_t scale_mode
) {
    int n = (cols + block_size - 1) / block_size;
    auto scales = torch::empty({rows*n}, data.options());
    auto packed = torch::empty({rows*cols}, data.options().dtype(torch::kUInt8));

    float host_table[128];
    for (int i = 0; i < 128; i++) {
        int e = (i >> 3) & 0xF;
        int m = i & 0x7;
        if (e == 15 && m == 7) host_table[i] = 448.0f;        
        else if (e == 0) host_table[i] = m * powf(2, -9);          
        else host_table[i] = (1.0f + m / 8.0f) * powf(2, e-7); 
    }
    cudaMemcpyToSymbol(_E4M3_TABLE, host_table, sizeof(host_table));

    //求一下最大值
    float gmax = 0.0f;
    gmax = data.abs().max().item<float>();

    //同样的一个线程处理一个block_size的元素
    dim3 block(256);
    dim3 grid(n, (rows + 255)/256, 1);
    quant_mxfp8_kernel<<<grid, block>>>(
        data.data_ptr<float>(),
        scales.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        (int)rows, (int)cols, (int)block_size, 
        (int)scale_mode, gmax
    );
    return {packed, scales};
}

torch::Tensor dequant_mxfp8(
    torch:: Tensor packed, 
    torch::Tensor scales, 
    int64_t rows, int64_t cols, 
    int64_t block_size, std::string& output_type
) {
    auto out = torch::empty({rows*cols}, scales.options().dtype(torch::kFloat32));
    dim3 block(256);
    dim3 grid((int)((cols + block_size - 1) / block_size),(rows + 255) / 256, 1);
    dequant_mxfp8_kernel<<<grid, block>>>(
        packed.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        scales.data_ptr<float>(),
        (int)rows, (int)cols, (int)block_size
    );

    if (output_type == "fp16") return out.to(torch::kHalf);      
    if (output_type == "bf16") return out.to(torch::kBFloat16);  
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant", &quant_mxfp8, "MXFP8 quantize -> (packed, scales)");
    m.def("dequant", &dequant_mxfp8, "MXFP8 dequantize -> dequant");
}