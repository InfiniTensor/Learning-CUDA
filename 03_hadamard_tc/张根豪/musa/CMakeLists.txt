cmake_minimum_required(VERSION 3.18)

# The MUSA toolkit ships its CMake language module separately from upstream
# CMake. Keep this project independent from the original CUDA build.
list(APPEND CMAKE_MODULE_PATH "/usr/local/musa-5.1.0/cmake")

# CMake 4 removed this old helper, which is still referenced by the toolkit's
# FindMUSA module. Recreate its cache variables before loading that module.
if(NOT COMMAND cmake_initialize_per_config_variable)
  function(cmake_initialize_per_config_variable prefix description)
    set(${prefix} "" CACHE STRING "${description}")
    foreach(config DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
      set(${prefix}_${config} "" CACHE STRING "${description}")
    endforeach()
  endfunction()
endif()

include(EnableMUSALanguage)

project(fast_hadamard_musa LANGUAGES CXX MUSA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MUSA_STANDARD 17)
set(CMAKE_MUSA_STANDARD_REQUIRED ON)

set(HADAMARD_WARPS_PER_BLOCK 4 CACHE STRING "Warp block size")
set(HADAMARD_FUSED_WARPS_PER_BLOCK ${HADAMARD_WARPS_PER_BLOCK} CACHE STRING
    "Fused INT4 warp block size")
set(HADAMARD_QUANT_VECTOR_STORE 0 CACHE STRING "INT4 wide store mode")

set(HADAMARD_DEFINITIONS
  HADAMARD_WARPS_PER_BLOCK=${HADAMARD_WARPS_PER_BLOCK}
  HADAMARD_FUSED_WARPS_PER_BLOCK=${HADAMARD_FUSED_WARPS_PER_BLOCK}
  HADAMARD_QUANT_VECTOR_STORE=${HADAMARD_QUANT_VECTOR_STORE})

find_library(MUSA_RUNTIME musart
  PATHS "/usr/local/musa-5.1.0/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(hadamard_bench
  src/main.mu
  src/hadamard.mu
  src/hadamard_warp.mu)

add_executable(hadamard_advanced_bench
  src/advanced_bench.mu
  src/hadamard.mu
  src/hadamard_warp.mu)

foreach(target hadamard_bench hadamard_advanced_bench)
  target_include_directories(${target} PRIVATE src)
  target_compile_definitions(${target} PRIVATE ${HADAMARD_DEFINITIONS})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:MUSA>:-O3>
    $<$<COMPILE_LANGUAGE:MUSA>:-include${CMAKE_CURRENT_SOURCE_DIR}/src/musa_compat.h>)
  target_link_libraries(${target} PRIVATE ${MUSA_RUNTIME})
endforeach()

set_target_properties(hadamard_bench hadamard_advanced_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})