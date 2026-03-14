cmake_minimum_required(VERSION 3.26)

project(hadacore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v3.4.0
)

FetchContent_MakeAvailable(cutlass)

add_executable(hadacore
        src/main.cu
        src/hadacore.cu
)

target_include_directories(hadacore PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(hadacore PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
)