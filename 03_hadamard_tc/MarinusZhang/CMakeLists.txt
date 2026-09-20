cmake_minimum_required(VERSION 3.24)
project(hw_hadamard_tc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Target GPU: RTX 4090 = Ada = sm_89.
#
# Do not rely on CMAKE_CUDA_ARCHITECTURES: CMake fills it with the nvcc default
# (sm_52 for CUDA 12.8), and nvcuda::wmma only exists for __CUDA_ARCH__ >= 700, so
# building for sm_52 makes every WMMA fragment fail to compile.
# Override with -DHW_CUDA_ARCH=90 or similar.
set(HW_CUDA_ARCH "89" CACHE STRING "CUDA compute capability to build for (89 = Ada)")
set(CMAKE_CUDA_ARCHITECTURES "${HW_CUDA_ARCH}")

find_package(CUDAToolkit REQUIRED)

# Core library: CPU FP32 reference plus GPU entry points.
add_library(hw_core STATIC
    src/hadamard_ref.cu
    src/hadamard_gpu.cu
)
target_include_directories(hw_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(hw_core PUBLIC CUDA::cudart)

set(HW_CUDA_FLAGS --generate-line-info -Xcompiler=-Wall)
target_compile_options(hw_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${HW_CUDA_FLAGS}>)

# Executables: toolchain probe, correctness harness, benchmark driver.
add_executable(hw_probe src/probe_main.cu)
add_executable(hw_tests tests/test_correctness.cu)
add_executable(hw_bench bench/bench_hadamard.cu)

foreach(target hw_probe hw_tests hw_bench)
    target_link_libraries(${target} PRIVATE hw_core)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${HW_CUDA_FLAGS}>)
endforeach()

enable_testing()
add_test(NAME hadamard_probe COMMAND hw_probe)
add_test(NAME hadamard_correctness COMMAND hw_tests)
