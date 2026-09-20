cmake_minimum_required(VERSION 3.24)

project(hadamard_cudacore LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(HADAMARD_SOURCES
    kernels/hadamard_api.cu
    kernels/hadamard_v0.cu
    kernels/hadamard_v1.cu
    kernels/hadamard_v2.cu
)

add_library(hadamard SHARED ${HADAMARD_SOURCES})
target_include_directories(hadamard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hadamard PUBLIC CUDA::cudart)
target_compile_options(hadamard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(hadamard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(benchmark_cuda benchmark/benchmark_cuda.cu)
target_include_directories(benchmark_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_cuda PRIVATE hadamard CUDA::cudart)
target_compile_options(benchmark_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME cuda_versions_smoke
        COMMAND benchmark_cuda --kernel all --batch 2 --dim 128
                --dtype fp16 --warmup 2 --repeat 5
                --csv ${CMAKE_CURRENT_BINARY_DIR}/smoke_results.csv
    )
endif()
