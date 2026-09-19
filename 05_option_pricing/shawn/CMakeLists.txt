cmake_minimum_required(VERSION 3.24)

project(derivatives_pricing LANGUAGES CXX)

find_package(Threads REQUIRED)

option(ENABLE_CUDA "Build the CUDA Monte Carlo backend when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pricing_core
    src/config.cpp
    src/pricing_cpu.cpp
)
target_include_directories(pricing_core PUBLIC include)
target_link_libraries(pricing_core PUBLIC Threads::Threads)
target_compile_options(pricing_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

set(PRICING_CUDA_ENABLED OFF)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(pricing_core PRIVATE src/pricing_cuda.cu)
        target_link_libraries(pricing_core PUBLIC CUDA::cudart CUDA::curand)
        target_compile_definitions(pricing_core PUBLIC PRICING_HAS_CUDA=1)
        set_target_properties(pricing_core PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET pricing_core PROPERTY CUDA_ARCHITECTURES 60 70 75 80)
        endif()
        set(PRICING_CUDA_ENABLED ON)
    endif()
endif()

if(NOT PRICING_CUDA_ENABLED)
    target_sources(pricing_core PRIVATE src/pricing_cuda_stub.cpp)
endif()

add_executable(derivatives_pricer src/main.cpp)
target_link_libraries(derivatives_pricer PRIVATE pricing_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(pricing_tests tests/test_pricing.cpp)
    target_link_libraries(pricing_tests PRIVATE pricing_core)
    target_compile_definitions(pricing_tests PRIVATE
        PRICING_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME pricing_tests COMMAND pricing_tests)
endif()

message(STATUS "CUDA pricing backend: ${PRICING_CUDA_ENABLED}")
