cmake_minimum_required(VERSION 3.24)

project(quant_dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(qd_core STATIC
    src/config.cpp
    src/io.cpp
    src/reference.cpp
    src/metrics.cpp
)
target_include_directories(qd_core PUBLIC include)
target_compile_options(qd_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(qd_cuda STATIC src/cuda_ops.cu)
target_include_directories(qd_cuda PUBLIC include)
target_link_libraries(qd_cuda PUBLIC qd_core)
target_compile_options(qd_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>
)

add_executable(quant_dequant src/main.cpp)
target_link_libraries(quant_dequant PRIVATE qd_cuda qd_core)
target_compile_options(quant_dequant PRIVATE -Wall -Wextra -Wpedantic)

add_executable(cuda_smoke src/cuda_smoke.cu)
target_compile_options(cuda_smoke PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>
)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_executable(test_formats tests/test_formats.cpp)
    target_include_directories(test_formats PRIVATE include)
    target_compile_options(test_formats PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME formats COMMAND test_formats)

    add_executable(test_reference tests/test_reference.cpp)
    target_link_libraries(test_reference PRIVATE qd_core)
    target_compile_options(test_reference PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME reference COMMAND test_reference)

    add_executable(test_io tests/test_io.cpp)
    target_link_libraries(test_io PRIVATE qd_core)
    target_compile_options(test_io PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME io COMMAND test_io)

    add_test(NAME cuda_smoke COMMAND cuda_smoke)
    set_tests_properties(cuda_smoke PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME gpu_pipeline
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_pipeline_test.py
                $<TARGET_FILE:quant_dequant>
                $<TARGET_FILE:cuda_smoke>
                ${CMAKE_SOURCE_DIR}/tools/generate_test_data.py
                ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(gpu_pipeline PROPERTIES DEPENDS cuda_smoke SKIP_RETURN_CODE 77)
endif()
