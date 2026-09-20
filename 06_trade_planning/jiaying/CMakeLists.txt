cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()
project(trade_flow LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_executable(trade_flow src/main.cpp src/graph.cpp src/cpu.cpp src/cuda.cu)
target_include_directories(trade_flow PRIVATE include)
target_compile_options(trade_flow PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME correctness COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py --exe $<TARGET_FILE:trade_flow>
  --output ${CMAKE_CURRENT_BINARY_DIR}/validation)
set_tests_properties(correctness PROPERTIES TIMEOUT 240)
