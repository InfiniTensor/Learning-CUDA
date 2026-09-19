cmake_minimum_required(VERSION 3.24)

project(quant_dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(low_precision STATIC
  src/low_precision.cu
)
target_include_directories(low_precision PUBLIC include)
target_link_libraries(low_precision PUBLIC CUDA::cudart)
target_compile_options(low_precision PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

add_executable(quant_dequant
  src/main.cu
)
target_link_libraries(quant_dequant PRIVATE low_precision CUDA::cudart)
target_compile_options(quant_dequant PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

enable_testing()
add_test(NAME quant_dequant_tests COMMAND quant_dequant --test)
