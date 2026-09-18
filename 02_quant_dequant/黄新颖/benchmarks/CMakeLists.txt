add_executable(quant_dequant_bench
    benchmark_main.cpp
)

target_link_libraries(quant_dequant_bench PRIVATE quant_dequant::core)

quant_dequant_enable_warnings(quant_dequant_bench)
