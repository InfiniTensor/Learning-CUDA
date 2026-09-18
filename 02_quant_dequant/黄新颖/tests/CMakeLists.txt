# 构建并注册一个可由 CTest 独立执行的项目测试模块。
#
# test_entry_main.cpp 通过编译定义调用对应测试源的 run_*_tests()。每个模块都链接
# 同一份核心静态库，却拥有独立进程、CTest 名称和标签；失败时 CTest 能精确报告
# 出错模块，且 ctest -L 可筛选 CUDA、格式或 I/O 类测试。
#
# 参数：
# - test_name：测试可执行文件名，也是 CTest 条目名；
# - test_source：定义唯一 run_*_tests() 函数的测试源文件；
# - test_entry：由测试入口文件调用的函数名；
# - ARGN：可选 CTest 标签列表。
function(add_quant_dequant_test test_name test_source test_entry)
    add_executable(${test_name}
        test_entry_main.cpp
        ${test_source}
    )

    target_link_libraries(${test_name} PRIVATE quant_dequant::core)

    # format codec 是内部实现；codec 和格式专用测试需要直接包含 src/formats
    # 下的 .cuh，因此所有测试目标统一获得 src 私有 include 路径。
    target_include_directories(${test_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
    )

    target_compile_definitions(${test_name} PRIVATE
        QUANT_DEQUANT_TEST_ENTRY=${test_entry}
    )
    quant_dequant_enable_warnings(${test_name})

    add_test(NAME ${test_name} COMMAND ${test_name})
    if (ARGN)
        set_tests_properties(${test_name} PROPERTIES LABELS "${ARGN}")
    endif()
endfunction()

# 纯 host 模块：不要求实际 CUDA device，适合任何开发环境快速运行。
add_quant_dequant_test(quant_dequant_config_tests
    test_config.cpp run_config_parser_tests "unit;config")
add_quant_dequant_test(quant_dequant_tensor_io_tests
    test_tensor_io.cpp run_tensor_io_tests "unit;io")
add_quant_dequant_test(quant_dequant_quantized_io_tests
    test_quantized_io.cpp run_quantized_io_tests "unit;io")
add_quant_dequant_test(quant_dequant_quantized_tensor_tests
    test_quantized_tensor.cpp run_quantized_tensor_tests "unit;model")
add_quant_dequant_test(quant_dequant_metrics_tests
    test_metrics.cpp run_metrics_tests "unit;metrics")
add_quant_dequant_test(quant_dequant_reference_dispatch_tests
    test_reference_dispatch.cpp run_reference_dispatch_tests "unit;reference")
add_quant_dequant_test(quant_dequant_mxfp8_reference_tests
    test_mxfp8_reference.cpp run_mxfp8_reference_tests "unit;reference;mxfp8")
add_quant_dequant_test(quant_dequant_nvfp4_reference_tests
    test_nvfp4_reference.cpp run_nvfp4_reference_tests "unit;reference;nvfp4")

# codec 测试会编译少量 device 函数；无 CUDA runtime 时其 device 子断言自行跳过。
add_quant_dequant_test(quant_dequant_mxfp8_codec_tests
    test_mxfp8_codec.cu run_mxfp8_codec_tests "unit;codec;mxfp8;cuda")
add_quant_dequant_test(quant_dequant_nvfp4_codec_tests
    test_nvfp4_codec.cu run_nvfp4_codec_tests "unit;codec;nvfp4;cuda")

# CUDA pipeline 测试在无 device/driver 时明确跳过数值 kernel 对照，而不是失败。
add_quant_dequant_test(quant_dequant_cuda_dispatch_tests
    test_cuda_dispatch.cpp run_cuda_dispatch_tests "integration;cuda")
add_quant_dequant_test(quant_dequant_device_quantized_tensor_tests
    test_device_quantized_tensor.cu run_device_quantized_tensor_tests "integration;cuda")
add_quant_dequant_test(quant_dequant_cuda_timer_tests
    test_cuda_timer.cu run_cuda_timer_tests "integration;cuda;common")
add_quant_dequant_test(quant_dequant_cuda_profile_tests
    test_cuda_profile.cu run_cuda_profile_tests "integration;cuda;profile;mxfp8")
add_quant_dequant_test(quant_dequant_mxfp8_cuda_tests
    test_mxfp8_cuda.cu run_mxfp8_cuda_tests "integration;cuda;mxfp8")
add_quant_dequant_test(quant_dequant_mxfp8_dequantize_cuda_tests
    test_mxfp8_dequantize_cuda.cu run_mxfp8_dequantize_cuda_tests
    "integration;cuda;mxfp8")
add_quant_dequant_test(quant_dequant_nvfp4_cuda_tests
    test_nvfp4_cuda.cu run_nvfp4_cuda_tests "integration;cuda;nvfp4")
add_quant_dequant_test(quant_dequant_nvfp4_dequantize_cuda_tests
    test_nvfp4_dequantize_cuda.cu run_nvfp4_dequantize_cuda_tests
    "integration;cuda;nvfp4")
