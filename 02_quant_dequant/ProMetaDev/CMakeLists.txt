cmake_minimum_required(VERSION 3.20)
project(lowprec_quant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 目标架构可被 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖；默认 90（PTX 可前向 JIT）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(lowprec
    src/main.cpp
    src/io.cpp
    src/metrics.cpp
    src/reference.cpp
    src/kernels.cu
)

target_include_directories(lowprec PRIVATE include)
target_link_libraries(lowprec PRIVATE CUDA::cudart)

set_target_properties(lowprec PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
