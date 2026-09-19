cmake_minimum_required(VERSION 3.24)

project(gpu_vector_search LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA search and IVF assignment backends when available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vector_search_core
    src/config.cpp
    src/file_io.cpp
    src/search_cpu.cpp
    src/ivf.cpp
)
target_include_directories(vector_search_core PUBLIC include)
target_compile_options(vector_search_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

set(VECTOR_SEARCH_CUDA_ENABLED OFF)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(vector_search_core PRIVATE src/search_cuda.cu)
        target_link_libraries(vector_search_core PUBLIC CUDA::cudart)
        target_compile_definitions(vector_search_core PUBLIC VECTOR_SEARCH_HAS_CUDA=1)
        set_target_properties(vector_search_core PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET vector_search_core PROPERTY
                CUDA_ARCHITECTURES 60 70 75 80)
        endif()
        set(VECTOR_SEARCH_CUDA_ENABLED ON)
    endif()
endif()

if(NOT VECTOR_SEARCH_CUDA_ENABLED)
    target_sources(vector_search_core PRIVATE src/search_cuda_stub.cpp)
endif()

add_executable(vector_search src/main.cpp)
target_link_libraries(vector_search PRIVATE vector_search_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(vector_search_tests tests/test_vector_search.cpp)
    target_link_libraries(vector_search_tests PRIVATE vector_search_core)
    target_compile_definitions(vector_search_tests PRIVATE
        VECTOR_SEARCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME vector_search_tests COMMAND vector_search_tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gpu_validation_tests
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_validation.py)
    endif()
endif()

message(STATUS "CUDA vector-search backend: ${VECTOR_SEARCH_CUDA_ENABLED}")
