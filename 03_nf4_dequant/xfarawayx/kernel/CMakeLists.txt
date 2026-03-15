cmake_minimum_required(VERSION 3.18)
project(nf4_dequant LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ---------- GPU 架构 ----------
# 用法:
#   cmake .. -DGPU_ARCH=80          # A100
#   cmake .. -DGPU_ARCH=89          # RTX 4090
#   cmake .. -DGPU_ARCH=90          # H100
#   cmake .. -DGPU_ARCH="80;89;90"  # 多架构
#   cmake ..                        # 自动检测
if(DEFINED GPU_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
    message(STATUS "GPU architecture (user-specified): ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # 自动检测当前 GPU 的 compute capability
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE _detected_arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _detect_result
    )
    if(_detect_result EQUAL 0 AND _detected_arch)
        # 取第一块 GPU，格式 "8.0" → "80"
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _first_arch "${_detected_arch}")
        string(REPLACE "." "" _arch_num "${_first_arch}")
        set(CMAKE_CUDA_ARCHITECTURES ${_arch_num})
        message(STATUS "GPU architecture (auto-detected): sm_${_arch_num}")
    else()
        # 检测失败时回退到常见架构
        set(CMAKE_CUDA_ARCHITECTURES "80;89;90")
        message(STATUS "GPU architecture (fallback): ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# ---------- 临时目录 ----------
set(NF4_LOCAL_TMP_DIR "${CMAKE_BINARY_DIR}/.tmp")
file(MAKE_DIRECTORY "${NF4_LOCAL_TMP_DIR}")
set(NF4_TMP_ENV_PREFIX
	"${CMAKE_COMMAND} -E env TMPDIR=${NF4_LOCAL_TMP_DIR} TMP=${NF4_LOCAL_TMP_DIR} TEMP=${NF4_LOCAL_TMP_DIR}")
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${NF4_TMP_ENV_PREFIX}")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${NF4_TMP_ENV_PREFIX}")

# ---------- 构建目标 ----------
add_executable(nf4_dequant main.cu)
target_include_directories(nf4_dequant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
