add_executable(quant_dequant
    main.cpp
)

target_link_libraries(quant_dequant PRIVATE quant_dequant::core)

quant_dequant_enable_warnings(quant_dequant)
