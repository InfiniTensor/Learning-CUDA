cmake_minimum_required(VERSION 3.20)
project(hadamard_fht LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

add_executable(hadamard
    src/main.cpp
    src/io.cpp
    src/cpu_ref.cpp
    src/fht.cu
)

target_include_directories(hadamard PRIVATE include)
target_link_libraries(hadamard PRIVATE CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hadamard PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP 已启用：CPU 参考实现将并行执行")
endif()

set_target_properties(hadamard PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
