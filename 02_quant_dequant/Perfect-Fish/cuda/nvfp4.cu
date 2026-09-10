#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <tuple>

__device__ const float _E2M1_TABLE[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

__device__ int _float_to_e2m1(float val) {
    if (val == 0) return 0;
    float a = fabsf(val);

    int idx = 0;
    if (a >= 5.0) idx = 7;
    else if (a >= 3.5) idx = 6;
    else if (a >= 2.5) idx = 5;
    else if (a >= 1.75) idx = 4;
    else if (a >= 1.25) idx = 3;
    else if (a >= 0.75) idx = 2;
    else if (a >= 0.25) idx = 1;
    else idx = 0;
    return (val > 0) ? idx : idx + 8;

}

__global__ void quant_nvfp4_kernel(
    const float* data, 
    uint8_t* packed, 
    float* scales, 
    const int rows, const int cols, 
    const int block_size, 
    const int tensor_mode
) {
    int tid = threadIdx.x;
    int r = blockIdx.y * blockDim.x + tid;
    int c = blockIdx.x;

    if (r >= rows) return;
    int stride = r * cols + c * block_size;
    int num = (cols + block_size -1) / block_size;

    float global_scale = 0.0f;
    if (tensor_mode) {
        // 感觉这里找全局最大值有点奇怪，好像可以优化，毕竟是并行运行的
        float gmax = 0.0f;
        for (int i = 0; i < rows * cols; i++) {
            if (gmax < fabsf(data[i])) gmax = fabsf(data[i]);
        }
        global_scale = (gmax > 0) ? (gmax / 6.0) : 1.0;
    } else {
        float gmax = 0.0f;
        for (int i = 0; i < block_size; i++) {
            if (gmax < fabsf(data[i+stride])) gmax = fabsf(data[i+stride]);
        }
        global_scale = (gmax > 0) ? (gmax / 6.0) : 1.0;
    }
    scales[r * num + c] = global_scale;

    // 量化 + 打包
    int half = cols / 2;
    for (int i = 0; i < block_size; i += 2) {
        float x0 = data[i+stride];
        float x1 = data[i+stride+1];
        int code0 = _float_to_e2m1(x0 / scales[r * num + c]);
        int code1 = _float_to_e2m1(x1 / scales[r * num + c]);
        packed[r*half + c*(block_size/2) + i/2] = (uint8_t)(code0 & 0xF) | ((code1 & 0xF) << 4);
    }
}


std::tuple<torch::Tensor, torch::Tensor> quant_nvfp4(
    torch::Tensor data, int64_t rows, int64_t cols,
    int64_t block_size, std::string& scale_mode
) {
    int num = (int)(cols + block_size -1) / block_size;
    int scale_num = (int)(rows * num);
    auto scales = torch::empty({scale_num}, data.options());
    // 打包的时候注意一个字节可以放两个元素
    auto packed = torch::empty({rows * (cols/2)}, data.options().dtype(torch::kUInt8));
    int tensor_mode = (scale_mode == "tensor") ? 1 : 0;

    // 一个线程一次性处理blocksize个元素
    dim3 block(256);
    dim3 grid(num, (rows + 255) / 256, 1);
    quant_nvfp4_kernel<<<grid, block>>>(
        data.data_ptr<float>(), 
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        (int)rows, (int)cols, 
        (int)block_size, tensor_mode
    );

    return {packed, scales};
}

__global__ void dequant_nvfp4_kernel( 
    const uint8_t* packed, 
    float* out,
    const float* scales, 
    const int rows, const int cols, 
    const int block_size
) {
    int tid = threadIdx.x;
    int r = blockIdx.y * blockDim.x + tid;
    int c = blockIdx.x;

    if (r >= rows) return;
    int stride = r * cols + c * block_size;
    int num = (cols + block_size -1) / block_size;
    int half = cols / 2;
    float s = scales[r*num + c];
    //拆包
    for (int i = 0; i < block_size; i += 2) {
        int codes = packed[r*half + c*(block_size/2) + i/2];
        int code0 = codes & 0xF;
        int code1 = (codes >> 4) & 0xF;
        out[i + stride] = _E2M1_TABLE[code0] * s;
        out[i + stride + 1] = _E2M1_TABLE[code1] * s;
    }
}

torch::Tensor dequant_nvfp4(
    torch::Tensor packed, 
    torch::Tensor scales, 
    int64_t rows, int64_t cols, 
    int64_t block_size, std::string& output_type
) {
    auto out = torch::empty({rows*cols}, scales.options().dtype(torch::kFloat32));
    dim3 block(256);
    dim3 grid((int)((cols + block_size - 1) / block_size),(rows + 255) / 256, 1);
    dequant_nvfp4_kernel<<<grid, block>>>(
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
    m.def("quant", &quant_nvfp4, "NVFP4 quantize -> (packed, scales)");
    m.def("dequant", &dequant_nvfp4, "NVFP4 dequantize -> dequant");
}