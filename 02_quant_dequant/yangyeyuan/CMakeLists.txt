cmake_minimum_required(VERSION 3.20)
project(low_precision_quant LANGUAGES CXX CUDA)

# 代码使用 C++17/CUDA 17；具体 GPU 架构由配置命令中的 CMAKE_CUDA_ARCHITECTURES
# 指定，避免把本机架构硬编码进工程。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(low_precision STATIC src/low_precision.cu)
# 头文件目录对库和 CLI 都公开，CLI 只依赖异步 host API。
target_include_directories(low_precision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(low_precision PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(low_precision PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 硬件/库对照路径（加分项）：默认关闭，关闭时构建产物与之前完全一致。
# 打开方式：cmake -DWITH_HW_PATH=ON（或 make WITH_HW_PATH=1）。
# cvt 路径只需要 CUDA 工具链自带的 cuda_fp8.h（sm_89+ 运行时可用）；
# gemm 路径还需要 cuBLASLt，通过 CUBLAS_ROOT 指定（如 pip 的 nvidia-cublas 包
# 或完整 CUDA 安装），找不到时只编译 cvt 路径并在 CLI 中报 gemm 不可用。
# Hadamard 旋转融合（方向三联动）：默认关闭，关闭时构建产物与之前完全一致。
# 打开方式：cmake -DWITH_HADAMARD=ON（或 make WITH_HADAMARD=1）。打开后相对路径
# 直接编译 ../../03_hadamard_tc/yangyeyuan/src/hadamard.cu（不复制源码），
# 配置文件中可设置 hadamard = "off|chunk128"；chunk128 在量化前对每个 128 列
# chunk 做归一化 FWHT 旋转，反量化后再变换一次作为逆旋转。
option(WITH_HADAMARD "build optional Hadamard rotation fusion" OFF)
if(WITH_HADAMARD)
  target_sources(low_precision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../03_hadamard_tc/yangyeyuan/src/hadamard.cu)
  target_include_directories(low_precision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../03_hadamard_tc/yangyeyuan/include)
  target_compile_definitions(low_precision PUBLIC LPQ_WITH_HADAMARD=1)
  message(STATUS "Hadamard fusion: chunk128 via ../../03_hadamard_tc/yangyeyuan")
endif()

option(WITH_HW_PATH "build optional hardware/library comparison path" OFF)
if(WITH_HW_PATH)
  target_sources(low_precision PRIVATE src/hw_path.cu)
  target_compile_definitions(low_precision PUBLIC LPQ_WITH_HW_PATH=1)
  set(CUBLAS_ROOT "" CACHE PATH "cuBLAS 安装根目录（含 include/ 与 lib/）")
  find_path(CUBLAS_INCLUDE_DIR cublasLt.h
    HINTS ${CUBLAS_ROOT}/include
          $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cublas/include)
  find_library(CUBLASLT_LIBRARY
    NAMES libcublasLt.so.12 libcublasLt.so cublasLt
    HINTS ${CUBLAS_ROOT}/lib
          $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cublas/lib)
  find_library(CUBLAS_LIBRARY
    NAMES libcublas.so.12 libcublas.so cublas
    HINTS ${CUBLAS_ROOT}/lib
          $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/cublas/lib)
  if(CUBLAS_INCLUDE_DIR AND CUBLASLT_LIBRARY AND CUBLAS_LIBRARY)
    target_compile_definitions(low_precision PUBLIC LPQ_HW_CUBLAS=1)
    target_include_directories(low_precision PRIVATE ${CUBLAS_INCLUDE_DIR})
    target_link_libraries(low_precision PUBLIC ${CUBLASLT_LIBRARY} ${CUBLAS_LIBRARY})
    message(STATUS "HW path: cvt + cuBLASLt gemm (${CUBLASLT_LIBRARY})")
  else()
    message(STATUS "HW path: cvt only (cuBLASLt not found; set CUBLAS_ROOT)")
  endif()
endif()

add_executable(low_precision_cli src/main.cu)
target_link_libraries(low_precision_cli PRIVATE low_precision)
target_compile_options(low_precision_cli PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

include(CTest)
if(BUILD_TESTING)
  # CTest 覆盖格式自测以及 FP16/BF16/奇数元素的端到端 CLI 测试。
  add_test(NAME low_precision_self_test COMMAND low_precision_cli --self-test)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(
    NAME low_precision_cli_test
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh
            $<TARGET_FILE:low_precision_cli> ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/cli_test)
  if(WITH_HADAMARD)
    # 仅在 Hadamard 融合构建中追加：验证 chunk128 旋转确实降低 NVFP4 误差。
    add_test(
      NAME low_precision_hadamard_test
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hadamard.sh
              $<TARGET_FILE:low_precision_cli> ${CMAKE_CURRENT_SOURCE_DIR}
              ${CMAKE_CURRENT_BINARY_DIR}/hadamard_test)
  endif()
endif()
