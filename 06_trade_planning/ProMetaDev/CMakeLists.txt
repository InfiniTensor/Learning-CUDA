cmake_minimum_required(VERSION 3.16)
project(maxflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 目标架构（默认 sm_90，可通过 -DSM_ARCH=90 覆盖）
# 同时生成 PTX，以便在更新架构的 GPU 上 JIT 编译运行
if(NOT DEFINED SM_ARCH)
    set(SM_ARCH 90)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH} -gencode arch=compute_${SM_ARCH},code=compute_${SM_ARCH}")

find_package(CUDA REQUIRED)
find_package(OpenMP)

add_executable(maxflow
    src/main.cpp
    src/io.cpp
    src/cpu_ref.cpp
    src/gpu_maxflow.cu
)

target_include_directories(maxflow PRIVATE include ${CUDA_INCLUDE_DIRS})
target_link_libraries(maxflow PRIVATE ${CUDA_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(maxflow PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP 已启用：CPU 参考实现将并行执行")
endif()

set_target_properties(maxflow PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
