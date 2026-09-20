cmake_minimum_required(VERSION 3.24)
project(fused_hadamard_quant LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_library(fused_hadamard_quant SHARED
    kernels/api.cu kernels/fused_v0.cu kernels/fused_v1.cu kernels/fused_v2.cu kernels/separate.cu)
target_include_directories(fused_hadamard_quant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(fused_hadamard_quant PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fused_hadamard_quant PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(benchmark_cuda benchmark/benchmark_cuda.cu)
target_link_libraries(benchmark_cuda PRIVATE fused_hadamard_quant)
target_include_directories(benchmark_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)
add_test(NAME correctness_smoke COMMAND benchmark_cuda --correctness-only --batch 4 --dim 128 --dtype all)
