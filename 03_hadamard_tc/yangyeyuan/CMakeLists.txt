cmake_minimum_required(VERSION 3.24)
project(hadamard_cuda LANGUAGES CXX CUDA)

# 工程使用 C++17；CUDA kernel 通过模板对 dtype 和 head_dim 做特化。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# 核心实现编译成静态库，命令行 benchmark 单独链接 cuBLAS。
add_library(hadamard STATIC src/hadamard.cu)
target_include_directories(hadamard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(hadamard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)
set_target_properties(hadamard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(hadamard_bench src/main.cu)
target_link_libraries(hadamard_bench PRIVATE hadamard CUDA::cublas)
target_compile_options(hadamard_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)

include(CTest)
if(BUILD_TESTING)
  # CTest 入口会执行覆盖全部维度、dtype 和量化格式的自测。
  add_test(NAME hadamard_correctness COMMAND hadamard_bench --self-test)
endif()
