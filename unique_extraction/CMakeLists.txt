cmake_minimum_required(VERSION 3.18)

project(unique_extraction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)

add_executable(unique_extraction src/unique_extraction.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(unique_extraction PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
endif()

target_link_libraries(unique_extraction OpenMP::OpenMP_CXX CUDA::cudart)

add_executable(generate_file tester/generate_file.cpp)
target_link_libraries(generate_file OpenMP::OpenMP_CXX)