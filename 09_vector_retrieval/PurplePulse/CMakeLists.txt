cmake_minimum_required(VERSION 3.24)

# 必须在 project() 之前设置，否则 CMake 可能先缓存一个旧架构默认值。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures")
endif()

project(purplepulse_vector_retrieval LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(vector_engine
  src/file_io.cpp
  src/cpu_search.cpp
  src/ivf_flat.cpp
  src/gpu_ivf_flat.cu
  src/gpu_search.cu
)
target_include_directories(vector_engine PUBLIC include)
target_compile_options(vector_engine PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(vector_search src/main.cpp)
target_link_libraries(vector_search PRIVATE vector_engine)

add_executable(ivf_build src/build_ivf.cpp)
target_link_libraries(ivf_build PRIVATE vector_engine)

add_executable(ivf_search src/search_ivf.cpp)
target_link_libraries(ivf_search PRIVATE vector_engine)

add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE vector_engine)

add_executable(file_io_tests tests/file_io_tests.cpp src/file_io.cpp)
target_include_directories(file_io_tests PRIVATE include)
target_compile_options(file_io_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ivf_flat_tests tests/ivf_flat_tests.cpp)
target_link_libraries(ivf_flat_tests PRIVATE vector_engine)

add_executable(stability_tests tests/stability_tests.cpp)
target_link_libraries(stability_tests PRIVATE vector_engine)

enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME file_io_tests COMMAND file_io_tests)
add_test(NAME ivf_flat_tests COMMAND ivf_flat_tests)
add_test(NAME stability_tests COMMAND stability_tests)
add_test(NAME vector_search_help COMMAND vector_search --help)
add_test(NAME ivf_build_help COMMAND ivf_build --help)
add_test(NAME ivf_search_help COMMAND ivf_search --help)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME benchmark_faiss_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_benchmark_faiss.py)
  add_test(NAME benchmark_ivf_sweep_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_benchmark_ivf_sweep.py)
  add_test(NAME benchmark_adaptive_nprobe_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_benchmark_adaptive_nprobe.py)
  add_test(NAME import_mnist_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_import_mnist.py)
  add_test(NAME import_sift1m_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_import_sift1m.py)
  add_test(NAME import_ann_benchmarks_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_import_ann_benchmarks.py)
  add_test(NAME plot_results_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_plot_results.py)
  add_test(NAME stage6_tools_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_stage6_tools.py)
  add_test(NAME benchmark_exact_breakdown_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_benchmark_exact_breakdown.py)
  add_test(NAME benchmark_exact_query_tile_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_benchmark_exact_query_tile.py)
  add_test(NAME stage9_tools_tests
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_stage9_tools.py)
endif()
