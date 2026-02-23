cmake_minimum_required(VERSION 3.22)
project(NF4_Dequant_CUDA LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(nf4_dequant
    main.cpp
    data_loader.cpp
    nf4_dequant_kernel.cu
)

target_include_directories(nf4_dequant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(nf4_dequant PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall>
)
