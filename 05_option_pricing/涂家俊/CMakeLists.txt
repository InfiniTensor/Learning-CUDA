# 配置 CUDA 定价器的构建目标、可选分析能力和自动化测试。
cmake_minimum_required(VERSION 3.24)

project(cuda_pricer LANGUAGES CXX CUDA)

# CUDA Toolkit 提供运行时、cuRAND 和 CUB。
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to the target GPU architecture.")
endif()

# 将分析能力和性能模式显式设为独立开关。
option(PRICER_ENABLE_NVTX "Enable NVTX performance markers" OFF)
option(PRICER_ENABLE_FAST_MATH "Enable CUDA fast math" OFF)

# 固定第三方库版本，保证干净环境能重复配置。
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# 与设备无关的金融模型、配置、统计和输出代码。
add_library(pricer_core STATIC
    src/black_scholes.cpp
    src/config.cpp
    src/cpu_pricer.cpp
    src/measurement.cpp
    src/output.cpp
    src/payoff.cpp
    src/statistics.cpp
    src/version.cpp
)
add_library(pricer::core ALIAS pricer_core)
target_compile_features(pricer_core PUBLIC cxx_std_17)
target_include_directories(pricer_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(pricer_core PUBLIC nlohmann_json::nlohmann_json)

# 单独构建设备端后端，避免 CPU 测试依赖 CUDA 内核实现细节。
add_library(pricer_cuda_backend STATIC src/cuda_backend.cu)
add_library(pricer::cuda_backend ALIAS pricer_cuda_backend)
target_compile_features(pricer_cuda_backend PUBLIC cxx_std_17)
target_include_directories(pricer_cuda_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(pricer_cuda_backend PUBLIC pricer::core CUDA::cudart)
if(MSVC)
    target_compile_options(pricer_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()
set_target_properties(pricer_cuda_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 快速数学只用于明确请求的性能实验，不能作为正确性默认值。
if(PRICER_ENABLE_FAST_MATH)
    target_compile_options(pricer_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# NVTX 标记仅为剖析提供时间线语义，常规构建不依赖它。
if(PRICER_ENABLE_NVTX)
    set(PRICER_NVTX_INCLUDE_DIR "" CACHE PATH
        "Directory containing nvtx3/nvToolsExt.h")
    if(NOT PRICER_NVTX_INCLUDE_DIR)
        unset(PRICER_NVTX_INCLUDE_DIR CACHE)
        find_path(PRICER_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS} "$ENV{NVTX_PATH}"
            PATH_SUFFIXES include)
    endif()
    if(NOT PRICER_NVTX_INCLUDE_DIR)
        message(FATAL_ERROR
            "PRICER_ENABLE_NVTX requires nvtx3/nvToolsExt.h; set PRICER_NVTX_INCLUDE_DIR.")
    endif()
    target_compile_definitions(pricer_cuda_backend PRIVATE PRICER_ENABLE_NVTX=1)
    target_include_directories(pricer_cuda_backend PRIVATE
        "${PRICER_NVTX_INCLUDE_DIR}")
    target_link_libraries(pricer_cuda_backend PRIVATE CUDA::nvtx3)
endif()

# CLI 负责把配置、定价后端和文件输出串成完整一次运行。
add_executable(pricer_cli src/main.cpp)
target_compile_features(pricer_cli PRIVATE cxx_std_17)
target_link_libraries(pricer_cli PRIVATE pricer::core pricer::cuda_backend)
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE PRICER_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(PRICER_GIT_COMMIT STREQUAL "")
    set(PRICER_GIT_COMMIT "unknown")
endif()
target_compile_definitions(pricer_cli PRIVATE
    PRICER_BUILD_TYPE="$<CONFIG>"
    PRICER_GIT_COMMIT="${PRICER_GIT_COMMIT}"
)
