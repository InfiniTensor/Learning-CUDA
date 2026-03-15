cmake_minimum_required(VERSION 3.20)
project(bilateral_filter CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include(CTest)

include_directories(${OpenCV_INCLUDE_DIRS} src)

# 主程序
add_executable(bilateral_filter
    src/main.cu
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(bilateral_filter ${OpenCV_LIBS})
set_target_properties(bilateral_filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 正确性测试
add_executable(test_correctness
    test/test_correctness.cpp
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(test_correctness ${OpenCV_LIBS})
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 性能测试
add_executable(test_performance
    test/test_performance.cpp
    src/bilateral_filter.cu
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_link_libraries(test_performance ${OpenCV_LIBS})
set_target_properties(test_performance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)
    add_test(NAME correctness COMMAND test_correctness)
    set_tests_properties(correctness PROPERTIES LABELS "correctness")

    add_test(NAME performance COMMAND test_performance)
    set_tests_properties(performance PROPERTIES LABELS "performance;benchmark")
endif()
