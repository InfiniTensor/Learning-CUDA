cmake_minimum_required(VERSION 3.18)
project(vsearch LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# GPU vector search engine (CUDA): exact baseline + IVF-Flat / IVF-PQ
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Covers common datacenter / desktop NVIDIA GPUs.
  # Override with: -DCMAKE_CUDA_ARCHITECTURES="80;86"
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89" CACHE STRING "CUDA architectures")
endif()

option(VSEARCH_BUILD_TESTS "Build host and GPU tests" ON)
option(VSEARCH_ENABLE_OMP "Enable OpenMP in host code when available" ON)

find_package(CUDAToolkit REQUIRED)

if(VSEARCH_ENABLE_OMP)
  find_package(OpenMP QUIET)
endif()

add_library(vsearch_core STATIC
  src/binary_io.cpp
  src/config.cpp
  src/cpu_reference.cpp
  src/result_io.cpp
  src/metric.cpp
  cuda/engine.cu
)

target_include_directories(vsearch_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vsearch_core PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(vsearch_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(vsearch_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(OpenMP_CXX_FOUND AND VSEARCH_ENABLE_OMP)
  target_link_libraries(vsearch_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(vsearch src/main.cpp)
target_link_libraries(vsearch PRIVATE vsearch_core)

if(VSEARCH_BUILD_TESTS)
  enable_testing()

  add_executable(test_host tests/test_host.cpp)
  target_link_libraries(test_host PRIVATE vsearch_core)
  add_test(NAME host_tests COMMAND test_host)

  add_executable(test_gpu tests/test_gpu.cu)
  target_link_libraries(test_gpu PRIVATE vsearch_core)
  add_test(NAME gpu_tests COMMAND test_gpu)
endif()
