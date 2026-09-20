cmake_minimum_required(VERSION 3.16)
project(nlm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 目标架构（默认 sm_90，可 -DSM_ARCH 覆盖）；同时生成 PTX 以兼容更新架构（如 sm_120）
if(NOT DEFINED SM_ARCH)
    set(SM_ARCH 90)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH} -gencode arch=compute_${SM_ARCH},code=compute_${SM_ARCH}")

find_package(CUDA REQUIRED)
find_package(OpenMP)

# OpenCV：用于图像读写（PNG/JPG）与题目要求的 fastNlMeansDenoising* 外部参考
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # 退化路径：尝试 pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_OPENCV opencv4)
    endif()
endif()
if(NOT OpenCV_FOUND AND NOT PC_OPENCV_FOUND)
    message(FATAL_ERROR "未找到 OpenCV。本项目用 OpenCV 做图像读写与参考实现对比，请先安装 libopencv-dev。")
endif()

add_executable(nlm
    src/main.cpp
    src/image_io.cpp
    src/cpu_ref.cpp
    src/nlm_gpu.cu
)

target_include_directories(nlm PRIVATE include ${CUDA_INCLUDE_DIRS})
target_link_libraries(nlm PRIVATE ${CUDA_LIBRARIES})

if(OpenCV_FOUND)
    target_include_directories(nlm PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(nlm PRIVATE ${OpenCV_LIBS})
    message(STATUS "OpenCV ${OpenCV_VERSION} 已启用（图像 I/O + 参考实现）")
else()
    target_include_directories(nlm PRIVATE ${PC_OPENCV_INCLUDE_DIRS})
    target_link_libraries(nlm PRIVATE ${PC_OPENCV_LIBRARIES})
    target_link_directories(nlm PRIVATE ${PC_OPENCV_LIBRARY_DIRS})
    message(STATUS "OpenCV 通过 pkg-config 启用")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(nlm PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP 已启用：CPU 参考实现并行执行")
endif()

set_target_properties(nlm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
