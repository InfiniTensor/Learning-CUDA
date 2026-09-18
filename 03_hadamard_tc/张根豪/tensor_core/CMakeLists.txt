# CMakeLists.txt — Tensor Core Hadamard 对照实验（与 Makefile 等价）
#
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build -j
#
# 注意：BF16 WMMA 需要 SM >= 80，因此这里不接受 7.x 架构。

cmake_minimum_required(VERSION 3.18)
project(hadamard_tensor_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
  string(REGEX MATCH "^[0-9]+" arch_number ${arch})
  if(arch_number LESS 80)
    message(FATAL_ERROR "Tensor Core module requires SM >= 80 (BF16 WMMA), got ${arch}")
  endif()
endforeach()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(HADAMARD_WARPS_PER_BLOCK 4 CACHE STRING "Warps/block: 2, 4, 8; 0=opt-in H200 shape preset")
set(HADAMARD_FUSED_WARPS_PER_BLOCK ${HADAMARD_WARPS_PER_BLOCK} CACHE STRING "Independent fused warps/block: 0, 2, 4, 8")
set(HADAMARD_QUANT_VECTOR_STORE 0 CACHE STRING "Opt-in 64/128-bit INT4 output stores for d=512/1024")
set(HADAMARD_TC_WARPS_PER_BLOCK 4 CACHE STRING "TC warps per block: 2, 4, 8")
set(HADAMARD_TC_VECTOR_SMEM 0 CACHE STRING "0=scalar; 1=vector shared; 2=vector shared and trim final warp barrier")

add_executable(hadamard_tc_bench
  src/tc_bench.cu
  src/hadamard_tc.cu
  ${PROJECT_ROOT}/src/hadamard.cu
  ${PROJECT_ROOT}/src/hadamard_warp.cu
)

target_include_directories(hadamard_tc_bench PRIVATE include ${PROJECT_ROOT}/include)
target_compile_definitions(hadamard_tc_bench PRIVATE
  HADAMARD_WARPS_PER_BLOCK=${HADAMARD_WARPS_PER_BLOCK}
  HADAMARD_FUSED_WARPS_PER_BLOCK=${HADAMARD_FUSED_WARPS_PER_BLOCK}
  HADAMARD_QUANT_VECTOR_STORE=${HADAMARD_QUANT_VECTOR_STORE}
  HADAMARD_TC_WARPS_PER_BLOCK=${HADAMARD_TC_WARPS_PER_BLOCK}
  HADAMARD_TC_VECTOR_SMEM=${HADAMARD_TC_VECTOR_SMEM})
target_compile_options(hadamard_tc_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(hadamard_tc_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
