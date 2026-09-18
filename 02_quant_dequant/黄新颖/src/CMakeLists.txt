add_library(quant_dequant_core STATIC
    quant_dequant.cpp
    config/config_parser.cpp
    io/tensor_io.cpp
    io/quantized_io.cpp
    metrics/metrics.cpp
    reference/reference_dispatch.cpp
    reference/mxfp8_reference.cpp
    reference/nvfp4_reference.cpp
    pipeline/quantize.cpp
    pipeline/dequantize.cpp
    cuda/mxfp8_quantize.cu
    cuda/mxfp8_dequantize.cu
    cuda/nvfp4_quantize.cu
    cuda/nvfp4_dequantize.cu
    common/cuda_stream.cu
    common/cuda_timer.cu
)

# 文件只承担 pipeline 编排，故保留 .cpp 后缀；但其中构造并写入
# thrust::device_vector，会实例化 CUDA 后端模板，必须交由 NVCC 编译。
# 它不定义任何 __global__ kernel；kernel 仍只位于 src/cuda/*.cu。
set_source_files_properties(pipeline/quantize.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(pipeline/dequantize.cpp PROPERTIES LANGUAGE CUDA)

add_library(quant_dequant::core ALIAS quant_dequant_core)

target_compile_features(quant_dequant_core PUBLIC cxx_std_20)

target_include_directories(quant_dequant_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(quant_dequant_core PUBLIC CUDA::cudart)

set_target_properties(quant_dequant_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

quant_dequant_enable_warnings(quant_dequant_core)
