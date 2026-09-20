cmake_minimum_required(VERSION 3.20)
project(gpu_vector_retrieval LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(GVS_BUILD_TESTS "Build correctness tests" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

add_library(gvs
    src/cpu_search.cpp
    src/file_io.cpp
    src/ivf_flat.cpp
    src/gpu_search.cu
)
target_include_directories(gvs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gvs PUBLIC CUDA::cudart)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gvs PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(gvs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gvs PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

add_executable(vector_search src/main.cpp)
target_link_libraries(vector_search PRIVATE gvs)

if(GVS_BUILD_TESTS)
    enable_testing()
    add_executable(gvs_tests tests/test_search.cpp)
    target_link_libraries(gvs_tests PRIVATE gvs)
    add_test(NAME gvs_correctness COMMAND gvs_tests)
endif()
