cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()
project(nbody LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(OpenMP REQUIRED)
add_executable(nbody src/main.cpp src/io.cpp src/cpu.cpp src/cuda.cu)
target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--ptxas-options=-v>)
# Avoid global fast-math: preserve finite-value checks and reference arithmetic.
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME physics_and_io COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate.py --exe $<TARGET_FILE:nbody>
  --output ${CMAKE_CURRENT_BINARY_DIR}/validation)
set_tests_properties(physics_and_io PROPERTIES TIMEOUT 180)
