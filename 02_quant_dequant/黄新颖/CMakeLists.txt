cmake_minimum_required(VERSION 3.22)

project(quant_dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake 3.22 尚未定义 CUDA20 对应的 NVCC 编译选项。host 侧 .cpp 继续
# 使用 C++20；.cu 文件使用 CUDA 12.1 与 CMake 3.22 均明确支持的 C++17。
# 当前 codec 和 kernel 约束在 C++17 可用特性内，以便两种编译路径一致。
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QUANT_DEQUANT_BUILD_TESTS "构建正确性测试" ON)
option(QUANT_DEQUANT_BUILD_APPS "构建命令行程序" ON)
option(QUANT_DEQUANT_BUILD_BENCHMARKS "构建性能评测程序" ON)
set(QUANT_DEQUANT_CUDA_ARCHITECTURES "" CACHE STRING
    "目标 CUDA 架构，例如 RTX 4060 使用 89；留空时由 CMake 决定")

if(QUANT_DEQUANT_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${QUANT_DEQUANT_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA 架构列表" FORCE)
endif()

include(CTest)
find_package(CUDAToolkit REQUIRED)

function(quant_dequant_enable_warnings target_name)
    target_compile_options(${target_name}
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endfunction()

# src 只生成可复用核心库；应用、测试和性能评测各自链接该库。
add_subdirectory(src)

if(QUANT_DEQUANT_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(BUILD_TESTING AND QUANT_DEQUANT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(QUANT_DEQUANT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
