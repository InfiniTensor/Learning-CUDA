add_rules("mode.debug", "mode.release")

target("nf4_dequantizer")
    set_kind("binary")
    add_files("main.cu", "src/dequantize.cu")
    
    -- 语言设置: C++17 和 CUDA
    set_languages("cxx17", "cuda")
    
    -- 目标 GPU 架构: T4 (75), A100 (80), 4090 (89)
    add_cugencodes("compute_75,sm_75")
    add_cugencodes("compute_80,sm_80")
    add_cugencodes("compute_89,sm_89")

    -- 编译选项
    if is_mode("release") then
        set_optimize("fastest") -- 对应 -O3
    end

    -- CUDA 特有标志
    -- -lineinfo: 生成行号信息，用于 Nsight Compute
    -- --ptxas-options=-v: 显示 PTX 汇编详细信息 (寄存器使用量等)
    -- -use_fast_math: 启用快速数学库
    add_cuflags("-lineinfo", "--ptxas-options=-v", "-use_fast_math")
    
    -- 头文件目录
    add_includedirs("src")
