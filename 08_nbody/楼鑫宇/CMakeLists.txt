cmake_minimum_required(VERSION 3.24)

set(NBODY_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA, MACA, or NONE")
set_property(CACHE NBODY_GPU_BACKEND PROPERTY STRINGS CUDA MACA NONE)
if(NBODY_GPU_BACKEND STREQUAL "CUDA")
    project(nbody_louxinyu LANGUAGES CXX CUDA)
elseif(NBODY_GPU_BACKEND STREQUAL "MACA" OR NBODY_GPU_BACKEND STREQUAL "NONE")
    project(nbody_louxinyu LANGUAGES CXX)
else()
    message(FATAL_ERROR "NBODY_GPU_BACKEND must be CUDA, MACA, or NONE")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NBODY_CORE_SOURCES
    src/config.cpp
    src/io.cpp
    src/cpu_solver.cpp)

if(NBODY_GPU_BACKEND STREQUAL "CUDA")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    find_package(CUDAToolkit REQUIRED)
    list(APPEND NBODY_CORE_SOURCES src/cuda_solver.cu)
elseif(NBODY_GPU_BACKEND STREQUAL "MACA")
    if(NOT CMAKE_CXX_COMPILER MATCHES "mxcc(\\.exe)?$")
        message(FATAL_ERROR
            "MACA build requires -DCMAKE_CXX_COMPILER=/opt/maca/mxgpu_llvm/bin/mxcc")
    endif()
    list(APPEND NBODY_CORE_SOURCES src/maca_solver.cpp)
    set_source_files_properties(src/maca_solver.cpp PROPERTIES
        COMPILE_OPTIONS "-x;maca;-offload-arch;native")
endif()

add_library(nbody_core ${NBODY_CORE_SOURCES})
target_include_directories(nbody_core PUBLIC include)
target_compile_options(nbody_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(NBODY_GPU_BACKEND STREQUAL "CUDA")
    target_link_libraries(nbody_core PUBLIC CUDA::cudart)
    target_compile_definitions(nbody_core PUBLIC NBODY_WITH_CUDA=1)
elseif(NBODY_GPU_BACKEND STREQUAL "MACA")
    target_compile_definitions(nbody_core PUBLIC NBODY_WITH_MACA=1)
endif()

add_executable(nbody src/main.cpp)
target_link_libraries(nbody PRIVATE nbody_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(nbody_tests tests/test_cpu.cpp)
    target_link_libraries(nbody_tests PRIVATE nbody_core)
    add_test(NAME cpu_core COMMAND nbody_tests)

    add_executable(nbody_io_tests tests/test_io.cpp)
    target_link_libraries(nbody_io_tests PRIVATE nbody_core)
    add_test(NAME io_parsing COMMAND nbody_io_tests)

    add_executable(nbody_trajectory_tests tests/test_trajectory.cpp)
    target_link_libraries(nbody_trajectory_tests PRIVATE nbody_core)
    add_test(NAME trajectory_roundtrip COMMAND nbody_trajectory_tests)

    add_executable(nbody_convergence_tests tests/test_convergence.cpp)
    target_link_libraries(nbody_convergence_tests PRIVATE nbody_core)
    add_test(NAME dt_convergence COMMAND nbody_convergence_tests)

    if(NBODY_GPU_BACKEND STREQUAL "CUDA")
        add_executable(nbody_cuda_tests tests/test_cuda_naive.cpp)
        target_link_libraries(nbody_cuda_tests PRIVATE nbody_core)
        add_test(NAME cuda_backends_vs_cpu COMMAND nbody_cuda_tests)

        add_executable(nbody_cuda_mixed_tests tests/test_cuda_mixed.cpp)
        target_link_libraries(nbody_cuda_mixed_tests PRIVATE nbody_core)
        add_test(NAME cuda_mixed_4096_vs_cpu COMMAND nbody_cuda_mixed_tests)

        add_executable(nbody_cuda_bh_tests tests/test_cuda_bh.cpp)
        target_link_libraries(nbody_cuda_bh_tests PRIVATE nbody_core)
        add_test(NAME cuda_bh_65536_100_steps_vs_direct COMMAND nbody_cuda_bh_tests)
    elseif(NBODY_GPU_BACKEND STREQUAL "MACA")
        add_executable(nbody_maca_tests tests/test_maca.cpp)
        target_link_libraries(nbody_maca_tests PRIVATE nbody_core)
        add_test(NAME maca_backends_vs_cpu COMMAND nbody_maca_tests)
    endif()
endif()
