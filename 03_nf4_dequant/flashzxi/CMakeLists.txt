cmake_minimum_required(VERSION 3.18)

project(nf4_dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nf4_dequant
        src/main.cu
        src/nf4_dequant_naive.cu
        src/nf4_dequant_warpN.cu
)

target_include_directories(nf4_dequant PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# 单 TU/简单工程：关闭 RDC 更利于调试
set_target_properties(nf4_dequant PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(nf4_dequant PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g -O0>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -O0>

        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>

        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CXX>>:-g -O2>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo -g -O2>
)