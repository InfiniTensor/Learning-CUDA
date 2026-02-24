cmake_minimum_required(VERSION 3.18)

project(cuda_demo LANGUAGES CXX CUDA)

# C++ / CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(nf4_dequant
        src/nf4_dequant.cu
        src/main.cpp
)
target_include_directories(nf4_dequant PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include
)

set_target_properties(nf4_dequant PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)