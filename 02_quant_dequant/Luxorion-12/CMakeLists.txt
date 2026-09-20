cmake_minimum_required(VERSION 3.20)
project(Luxorion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LUXORION_ENABLE_CUDA "Build gpu_baseline and its tests" OFF)
option(LUXORION_BUILD_HISTORY
  "Build the v0-v8 optimization-history executables" OFF)
if(LUXORION_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

function(luxorion_configure_target target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif()
endfunction()

# CPU reference：独立、可直接运行，不依赖CUDA工具链或GPU。
add_library(cpu_reference_lib STATIC
  cpu_reference/src/formats.cpp
  cpu_reference/src/io.cpp
  cpu_reference/src/metrics.cpp
  cpu_reference/src/mxfp8_cpu.cpp
  cpu_reference/src/nvfp4_cpu.cpp
  cpu_reference/src/quant_config.cpp
)
target_include_directories(cpu_reference_lib PUBLIC cpu_reference/include)
luxorion_configure_target(cpu_reference_lib)

add_executable(cpu_reference cpu_reference/src/main.cpp)
target_link_libraries(cpu_reference PRIVATE cpu_reference_lib)
luxorion_configure_target(cpu_reference)

enable_testing()
foreach(cpu_test IN ITEMS formats mxfp8 nvfp4 metrics io quant_config)
  add_executable(test_${cpu_test} cpu_reference/tests/test_${cpu_test}.cpp)
  target_link_libraries(test_${cpu_test} PRIVATE cpu_reference_lib)
  luxorion_configure_target(test_${cpu_test})
  add_test(NAME ${cpu_test} COMMAND test_${cpu_test})
endforeach()

if(LUXORION_ENABLE_CUDA)
  # GPU baseline只保存CUDA实现；CPU reference作为独立oracle参与测试。
  add_library(gpu_baseline STATIC
    gpu_baseline/src/mxfp8.cu
    gpu_baseline/src/nvfp4.cu
  )
  target_include_directories(gpu_baseline PUBLIC
    gpu_baseline/include
    cpu_reference/include
  )
  luxorion_configure_target(gpu_baseline)

  add_executable(generate_gpu_baseline_inputs
    gpu_baseline/tools/generate_inputs.cpp)
  target_link_libraries(generate_gpu_baseline_inputs PRIVATE cpu_reference_lib)
  luxorion_configure_target(generate_gpu_baseline_inputs)

  add_executable(gpu_baseline_benchmark
    gpu_baseline/tools/benchmark_main.cpp)
  target_link_libraries(gpu_baseline_benchmark PRIVATE
    gpu_baseline cpu_reference_lib)
  luxorion_configure_target(gpu_baseline_benchmark)

  add_executable(luxorion_cli gpu_baseline/tools/luxorion_cli.cu)
  target_link_libraries(luxorion_cli PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_cli PRIVATE gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_cli)
  add_test(NAME cli_mxfp8_block_fp16
    COMMAND luxorion_cli run
      ${CMAKE_CURRENT_SOURCE_DIR}/input/benchmark/normal_fp16_1024x1024_seed1234.tensor.bin
      ${CMAKE_CURRENT_SOURCE_DIR}/input/quant.cfg
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_mxfp8_block 0 1)
  add_test(NAME cli_mxfp8_tensor_stochastic_bf16
    COMMAND luxorion_cli run
      ${CMAKE_CURRENT_SOURCE_DIR}/input/benchmark/tail_normal_fp32_1023x1025_seed1234.tensor.bin
      ${CMAKE_CURRENT_SOURCE_DIR}/input/mxfp8_tensor_stochastic_bf16.cfg
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_mxfp8_tensor 0 1)
  add_test(NAME cli_nvfp4_block_fp32
    COMMAND luxorion_cli run
      ${CMAKE_CURRENT_SOURCE_DIR}/input/benchmark/outliers_fp32_1024x1024_seed1234.tensor.bin
      ${CMAKE_CURRENT_SOURCE_DIR}/input/nvfp4_block_nearest_fp32.cfg
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_nvfp4_block 0 1)
  add_test(NAME cli_nvfp4_tensor_stochastic_fp16
    COMMAND luxorion_cli run
      ${CMAKE_CURRENT_SOURCE_DIR}/input/benchmark/tail_normal_fp16_1023x1025_seed1234.tensor.bin
      ${CMAKE_CURRENT_SOURCE_DIR}/input/nvfp4_tensor_stochastic_fp16.cfg
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_nvfp4_tensor 0 1)
  add_test(NAME cli_independent_dequantize
    COMMAND luxorion_cli dequantize
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_nvfp4_tensor/weights.luxq
      ${CMAKE_CURRENT_BINARY_DIR}/cli_test_nvfp4_tensor/restored_again.tensor.bin 0 1)
  set_tests_properties(cli_independent_dequantize PROPERTIES
    DEPENDS cli_nvfp4_tensor_stochastic_fp16)
  set_tests_properties(
    cli_mxfp8_block_fp16 cli_mxfp8_tensor_stochastic_bf16
    cli_nvfp4_block_fp32 cli_nvfp4_tensor_stochastic_fp16
    cli_independent_dequantize PROPERTIES RUN_SERIAL TRUE)

  if(LUXORION_BUILD_HISTORY)
  add_executable(luxorion_v0
    optimization/v0_baseline/luxorion_cli.cu)
  target_link_libraries(luxorion_v0 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v0 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v0)

  foreach(tpb IN ITEMS 64 128 256 512)
    set(v1_target luxorion_v1_tpb${tpb})
    add_executable(${v1_target}
      optimization/v1_thread_block/luxorion_cli.cu)
    target_link_libraries(${v1_target} PRIVATE cpu_reference_lib)
    target_include_directories(${v1_target} PRIVATE
      gpu_baseline/src gpu_baseline/include)
    target_compile_definitions(${v1_target} PRIVATE
      LUX_DEQUANT_THREADS=${tpb})
    luxorion_configure_target(${v1_target})
  endforeach()

  add_executable(luxorion_v1
    optimization/v1_thread_block/luxorion_cli.cu)
  target_link_libraries(luxorion_v1 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v1 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v1)

  add_executable(luxorion_v2
    optimization/v2_coalescing/luxorion_cli.cu)
  target_link_libraries(luxorion_v2 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v2 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v2)

  add_executable(luxorion_v3
    optimization/v3_vectorized_memory/luxorion_cli.cu)
  target_link_libraries(luxorion_v3 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v3 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v3)

  add_executable(luxorion_v4
    optimization/v4_warp_shuffle/luxorion_cli.cu)
  target_link_libraries(luxorion_v4 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v4 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v4)

  add_executable(luxorion_v5
    optimization/v5_shared_reuse/luxorion_cli.cu)
  target_link_libraries(luxorion_v5 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v5 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v5)

  add_executable(luxorion_v6
    optimization/v6_bank_conflict/luxorion_cli.cu)
  target_link_libraries(luxorion_v6 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v6 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v6)

  add_executable(luxorion_v7
    optimization/v7_branch_reduction/luxorion_cli.cu)
  target_link_libraries(luxorion_v7 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v7 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v7)

  add_executable(luxorion_v8
    optimization/v8_ilp_unroll/luxorion_cli.cu)
  target_link_libraries(luxorion_v8 PRIVATE cpu_reference_lib)
  target_include_directories(luxorion_v8 PRIVATE
    gpu_baseline/src gpu_baseline/include)
  luxorion_configure_target(luxorion_v8)
  endif()

  foreach(gpu_test IN ITEMS mxfp8 nvfp4)
    add_executable(test_${gpu_test}_cuda
      gpu_baseline/tests/test_${gpu_test}_cuda.cpp)
    target_link_libraries(test_${gpu_test}_cuda PRIVATE
      gpu_baseline cpu_reference_lib)
    luxorion_configure_target(test_${gpu_test}_cuda)
    add_test(NAME ${gpu_test}_cuda COMMAND test_${gpu_test}_cuda)
    set_tests_properties(${gpu_test}_cuda PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()
