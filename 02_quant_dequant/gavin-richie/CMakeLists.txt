cmake_minimum_required(VERSION 3.18)
project(lowp_quant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default target: RTX 4060 (Ada, sm_89). Override per GPU:
#   3090=86  4060/4090=89  H20=90  5090=120
#   e.g. cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(lowp_core STATIC
  src/quant_kernels.cu
  src/file_io.cpp
)
target_include_directories(lowp_core PUBLIC include)

add_executable(lowp src/main.cu)
target_link_libraries(lowp PRIVATE lowp_core)

enable_testing()

function(lowp_test name)
  add_executable(${name} tests/${name}.cu)
  target_link_libraries(${name} PRIVATE lowp_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

lowp_test(test_formats)
lowp_test(test_io)
lowp_test(test_roundtrip)

# Optional hardware-accelerated comparison path (Transformer Engine, etc.).
# The Python comparison lives in scripts/te_compare.py (no C++ changes
# needed); this switch stays reserved for a future C++ TE linkage.
option(ENABLE_TE_COMPARISON "Build Transformer Engine comparison path" OFF)

# Optional check that the software codecs are bit-identical to the CUDA Math
# API conversion intrinsics (cuda_fp8.h / cuda_fp4.h). On Blackwell these map
# to the native cvt instructions, which only exist in arch-specific builds:
# the target rewrites e.g. 120 -> 120a. Needs CUDA >= 12.8 (cuda_fp4.h).
option(ENABLE_NATIVE_CVT_TEST
  "Build test_native_cvt (software codecs vs __nv_cvt intrinsics)" OFF)
if(ENABLE_NATIVE_CVT_TEST)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "ENABLE_NATIVE_CVT_TEST needs CUDA >= 12.8 (cuda_fp4.h)")
  endif()
  set(NATIVE_TEST_ARCHS "")
  foreach(a ${CMAKE_CUDA_ARCHITECTURES})
    if(a MATCHES "^[0-9]+$" AND a GREATER_EQUAL 100)
      list(APPEND NATIVE_TEST_ARCHS "${a}a")
    else()
      list(APPEND NATIVE_TEST_ARCHS "${a}")
    endif()
  endforeach()
  add_executable(test_native_cvt tests/test_native_cvt.cu)
  target_link_libraries(test_native_cvt PRIVATE lowp_core)
  set_target_properties(test_native_cvt PROPERTIES
    CUDA_ARCHITECTURES "${NATIVE_TEST_ARCHS}")
  add_test(NAME test_native_cvt COMMAND test_native_cvt)
endif()
