# CMakeLists.txt — 标准 CMake 构建（推荐方式）
#
# 构建命令（与训练营要求一致）：
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# 可选：
#   -DCMAKE_CUDA_ARCHITECTURES=86    指定 GPU 架构（默认 86，RTX 3060 Laptop）
#   -DHADAMARD_ENABLE_OPENMP=ON      为 CPU 参考实现启用 OpenMP 并行（默认关）

cmake_minimum_required(VERSION 3.18)
project(hadamard_tc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

option(HADAMARD_ENABLE_OPENMP "Enable OpenMP for CPU reference" OFF)
set(HADAMARD_WARPS_PER_BLOCK 4 CACHE STRING "Warp block size: 0 adaptive, 2, 4, 8")
set(HADAMARD_FUSED_WARPS_PER_BLOCK ${HADAMARD_WARPS_PER_BLOCK} CACHE STRING "INT4 warp block size: 0 adaptive, 2, 4, 8")
set(HADAMARD_QUANT_VECTOR_STORE 0 CACHE STRING "INT4 wide store: 0 off, 1 all supported, 2 d1024 only")
foreach(name HADAMARD_WARPS_PER_BLOCK HADAMARD_FUSED_WARPS_PER_BLOCK)
  if(NOT "${${name}}" MATCHES "^(0|2|4|8)$")
    message(FATAL_ERROR "${name} must be 0, 2, 4 or 8")
  endif()
endforeach()
if(NOT "${HADAMARD_QUANT_VECTOR_STORE}" MATCHES "^[012]$")
  message(FATAL_ERROR "HADAMARD_QUANT_VECTOR_STORE must be 0, 1 or 2")
endif()

add_executable(hadamard_bench
  src/main.cu
  src/hadamard.cu
  src/hadamard_warp.cu
)

add_executable(hadamard_advanced_bench
  src/advanced_bench.cu
  src/hadamard.cu
  src/hadamard_warp.cu
)

target_include_directories(hadamard_bench PRIVATE include)
target_include_directories(hadamard_advanced_bench PRIVATE include)
foreach(target hadamard_bench hadamard_advanced_bench)
  target_compile_definitions(${target} PRIVATE
    HADAMARD_WARPS_PER_BLOCK=${HADAMARD_WARPS_PER_BLOCK}
    HADAMARD_FUSED_WARPS_PER_BLOCK=${HADAMARD_FUSED_WARPS_PER_BLOCK}
    HADAMARD_QUANT_VECTOR_STORE=${HADAMARD_QUANT_VECTOR_STORE})
endforeach()

target_compile_options(hadamard_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_compile_options(hadamard_advanced_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

if(HADAMARD_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(hadamard_bench PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(hadamard_bench PRIVATE HADAMARD_OPENMP=1)
endif()

# 输出放到构建目录根部，便于 ./hadamard_bench 直接运行
set_target_properties(hadamard_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_target_properties(hadamard_advanced_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
