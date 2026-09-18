cmake_minimum_required(VERSION 3.18)
set(LP_BACKEND "CUDA" CACHE STRING "Device backend: CUDA or MUSA")
set_property(CACHE LP_BACKEND PROPERTY STRINGS CUDA MUSA)
if(LP_BACKEND STREQUAL "MUSA")
  set(MUSA_ROOT "/usr/local/musa" CACHE PATH "MUSA SDK directory")
  set(MUSA_ARCH "mp_22" CACHE STRING "MUSA architecture (S4000: mp_22)")
  set(CMAKE_CXX_COMPILER "${MUSA_ROOT}/bin/mcc")
  project(low_precision_core LANGUAGES CXX)
  add_compile_definitions(LP_BACKEND_MUSA)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/backends/musa/include")
  include_directories("${MUSA_ROOT}/include")
  link_directories("${MUSA_ROOT}/lib")
  link_libraries(musart)
  set(CMAKE_BUILD_RPATH "${MUSA_ROOT}/lib")
  file(GLOB_RECURSE MUSA_DEVICE_SOURCES CONFIGURE_DEPENDS
       app/*.cu runtime/*.cu benchmarks/*.cu tests/*.cu)
  set_source_files_properties(${MUSA_DEVICE_SOURCES} PROPERTIES LANGUAGE CXX
      COMPILE_FLAGS "-x musa --offload-arch=${MUSA_ARCH} -ffp-contract=off -gline-tables-only")
elseif(LP_BACKEND STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  project(low_precision_core LANGUAGES CXX CUDA)
else()
  message(FATAL_ERROR "LP_BACKEND must be CUDA or MUSA")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  find_program(SHA256SUM_EXECUTABLE NAMES sha256sum REQUIRED)
  add_test(NAME core_tools COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tools.py)
  add_executable(reduction_test tests/reduction.cu)
  target_include_directories(reduction_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(reduction_test PRIVATE TEST_NVFP4)
  target_compile_options(reduction_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  add_test(NAME core_reduction COMMAND reduction_test)
  add_executable(device_math_test tests/device_math.cu)
  target_include_directories(device_math_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME core_device_math COMMAND device_math_test)
  if(LP_BACKEND STREQUAL "MUSA")
    add_executable(musa_shuffle_test tests/musa_shuffle.cu)
    target_include_directories(musa_shuffle_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME musa_shuffle COMMAND musa_shuffle_test)
  endif()
endif()

# 每个模块独立编译；只有 runtime/workspace.cu 包含正式 kernel 定义。
# reference 位于 Core 内，普通运行也使用它校验结果，不再包含目录外源码。
set(CORE_SOURCES
  app/main.cpp
  app/run.cu
  runtime/workspace.cu
  benchmarks/benchmark.cu
  reference/oracle.cpp
  reference/frozen.cpp
)

# 同一套源码按格式编译两次，保留现有可执行名称与 Python 调用协议。
foreach(FORMAT IN ITEMS mxfp8 nvfp4)
  set(TARGET pipeline_${FORMAT})
  add_executable(${TARGET} ${CORE_SOURCES})
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(FORMAT STREQUAL "nvfp4")
    target_compile_definitions(${TARGET} PRIVATE TEST_NVFP4)
  endif()

  if(BUILD_TESTING)
    set(CPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/reference/${FORMAT}")
    target_sources(${TARGET} PRIVATE tests/regression.cu)
    target_compile_definitions(${TARGET} PRIVATE LP_ENABLE_TESTS LP_REFERENCE_DIR="${CPU_DIR}")
    add_test(NAME ${FORMAT}_pipeline_io
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pipeline.py
              ${FORMAT} $<TARGET_FILE:${TARGET}>)
    add_test(NAME ${FORMAT}_pipeline_regression
      COMMAND ${TARGET} --self-test ${CPU_DIR})
    add_test(NAME ${FORMAT}_frozen_hashes
      COMMAND ${SHA256SUM_EXECUTABLE} -c tests/golden/SHA256SUMS.txt)
    set_tests_properties(${FORMAT}_frozen_hashes PROPERTIES WORKING_DIRECTORY "${CPU_DIR}")
  endif()
endforeach()
