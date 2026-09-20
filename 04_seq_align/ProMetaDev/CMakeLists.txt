cmake_minimum_required(VERSION 3.16)
project(seqalign LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 目标架构（默认 sm_90，可通过 -DSM_ARCH 覆盖）
# 同时生成 PTX，以便在更新架构（如 sm_120）的 GPU 上由驱动 JIT 运行
if(NOT DEFINED SM_ARCH)
    set(SM_ARCH 90)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH} -gencode arch=compute_${SM_ARCH},code=compute_${SM_ARCH}")

find_package(CUDA REQUIRED)
find_package(OpenMP)

set(SA_SOURCES
    src/main.cpp
    src/io.cpp
    src/cpu_ref.cpp
    src/cpu_seeded.cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/kmer_index.cu")
    list(APPEND SA_SOURCES src/kmer_index.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/align_gpu.cu")
    list(APPEND SA_SOURCES src/align_gpu.cu)
endif()

add_executable(seqalign ${SA_SOURCES})

target_include_directories(seqalign PRIVATE include ${CUDA_INCLUDE_DIRS})
target_link_libraries(seqalign PRIVATE ${CUDA_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP 已启用：CPU 参考实现将并行执行")
endif()

set_target_properties(seqalign PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
