cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CudaPricingProject CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CUDA 架构 (gencode 方式, 兼容 CMake 3.16) ----
# 必须包含 compute_90 的 PTX, 让 CUDA 13 驱动 JIT 编译到 Blackwell(sm_120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90")

# ---- 优化 ----
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(cuda_pricing
    src/main.cpp
    src/bs_formula.cpp
    src/file_io.cpp
    src/pricing_kernels.cu
)
target_link_libraries(cuda_pricing PRIVATE curand)

install(TARGETS cuda_pricing RUNTIME DESTINATION bin)
