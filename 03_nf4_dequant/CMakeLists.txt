cmake_minimum_required(VERSION 3.18)

# 1. 项目与语言
project(nf4_dequantizer LANGUAGES CXX CUDA)

# 2. 版本与架构设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 目标架构：T4 (75), A100 (80), 4090 (89)
set(CMAKE_CUDA_ARCHITECTURES 75 80 89)

# 3. 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 4. 可执行文件
add_executable(nf4_dequantizer main.cu src/dequantize.cu)

# 5. 头文件目录
target_include_directories(nf4_dequantizer PRIVATE src)

# 6. 编译优化选项 (Release 模式)
# -O3 为最高级别优化 (Release 默认包含 -O3，但我们可以显式保证)
target_compile_options(nf4_dequantizer PRIVATE 
    $<$<CONFIG:Release>:-O3>
)

# 7. 性能分析与优化 (CUDA)
# -lineinfo: 生成行号信息，用于 Nsight Compute 对照源码
# --ptxas-options=-v: 显示 PTX 汇编详细信息 (如寄存器使用量)
# -use_fast_math: 启用快速数学库
target_compile_options(nf4_dequantizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --ptxas-options=-v
        -use_fast_math
        -O3 
    >
)

# 8. 链接选项 (如有必要)
# target_link_libraries(nf4_dequantizer PRIVATE ...)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
