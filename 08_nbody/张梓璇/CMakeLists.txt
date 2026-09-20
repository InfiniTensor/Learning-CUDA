cmake_minimum_required(VERSION 3.20)
project(nbody_cuda LANGUAGES CXX CUDA)

option(NBODY_FAST_MATH "Compile CUDA kernels with --use_fast_math" OFF)
option(NBODY_PTXAS_INFO "Print ptxas register/shared-memory information" OFF)
option(ENABLE_NVTX "Add lightweight NVTX ranges for Nsight Systems" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认覆盖训练营里常见的 Turing/Ampere 显卡，也可以在配置时自行覆盖：
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING "CUDA architectures" FORCE)
endif()

add_executable(nbody
    src/main.cpp
    src/nbody.cu
    src/cpu_reference.cpp
    src/io.cpp
)
target_include_directories(nbody PRIVATE src)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nbody PROPERTY CUDA_RUNTIME_LIBRARY Shared)

if(NBODY_FAST_MATH)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
if(NBODY_PTXAS_INFO)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
if(ENABLE_NVTX)
    target_compile_definitions(nbody PRIVATE NBODY_ENABLE_NVTX=1)
endif()

if(MSVC)
    target_compile_options(nbody PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4;-Xcompiler=/utf-8>
    )
else()
    target_compile_options(nbody PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME nbody_gpu_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_test.py
                $<TARGET_FILE:nbody> ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
