cmake_minimum_required(VERSION 3.24)
project(vector_search_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(vector_search STATIC
  src/io.cpp
  src/cpu_reference.cpp
  src/ivf_flat.cpp
  src/search.cu)
target_include_directories(vector_search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vector_search PUBLIC CUDA::cudart)
target_compile_options(vector_search PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(vector_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(vector_search_cli src/main.cpp)
target_link_libraries(vector_search_cli PRIVATE vector_search)
target_compile_options(vector_search_cli PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME vector_search_self_test COMMAND vector_search_cli self-test)
  add_test(NAME vector_search_cli_test
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh $<TARGET_FILE:vector_search_cli>)
endif()
