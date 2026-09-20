cmake_minimum_required(VERSION 3.18)
project(nbody_simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# 目标架构
#   61 = MX330（本地开发机，Pascal），用于正确性验证
#   89 = RTX 4090D（远程基准机，Ada），性能数字全部来自这一档
# 同时编译两档，保证本地能抓到 sm_89 的编译期错误，不必等上传后才发现。
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61 89)
endif()

# FP64 开关：只在小 N 验证时开启，用来区分算法误差与浮点误差。
option(NBODY_FP64 "Use double precision for force computation" OFF)
# fast-math 开关：V3 档要量化它的收益与精度代价，所以做成可切换而非常开。
option(NBODY_FAST_MATH "Enable -use_fast_math for CUDA" OFF)


find_package(OpenMP)

# ---------------------------------------------------------------------------
# 公共编译选项。MSVC 与 gcc/clang 的标志完全不同，分开处理。
# ---------------------------------------------------------------------------
function(nbody_apply_host_flags target)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(NBODY_FP64)
    target_compile_definitions(${target} PRIVATE NBODY_FP64)
  endif()
  if(MSVC)
    # /utf-8 是必须的，不是可选项：源码里的中文注释是 UTF-8，
    # 而 MSVC 默认按系统代码页（简体中文环境下是 936/GBK）解释源文件。
    # 误解码会吞掉注释后面的真实代码字符，产生一堆莫名其妙的
    # "未声明的标识符" 错误（伴随 warning C4819）。
    # _CRT_SECURE_NO_WARNINGS：我们刻意用 std::fopen 而非 fopen_s，
    # 后者不可移植（gcc 没有），而这里没有任何安全隐患。
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /W4 /permissive->)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
  endif()
endfunction()

# CPU 基线的 kScalarO3 / kOpenMP 两档依赖自动向量化，
# 所以要显式开 -O3 -march=native（gcc）或 /O2 /arch:AVX2（MSVC）。
# 注意 -march=native 在远程机上编译时会针对那台 CPU 优化，
# 这正是我们想要的——基线必须是"同级别实现"，不能自缚手脚。
#
# 刻意不加 -ffast-math / /fp:fast：这些文件不止用于 CPU 基线的计时，
# 还被两个更要紧的用途复用——
#   1. diagnostics.cpp 是能量/动量守恒判据的"标准答案"，
#      validate.sh 里"leapfrog energy error < 1e-3"这类 PASS/FAIL 直接
#      依赖它的数值；
#   2. cpu_reference.cpp 的 kOpenMP 档同时被 main.cu 的 CrossCheckAcc
#      当作 GPU 加速度的比对基准。
# fast-math 允许编译器重排求和顺序、启用近似倒数指令，这会让"标准答案"
# 本身带上不受控的误差，而这两处恰恰是用来裁定"GPU 算得对不对"的地方——
# 拿一个放宽了精度语义的参考去评判 GPU，验证结果就不可信。
# GPU 侧的 fast-math 收益与代价是 V3 档要专门测量、单独开关的问题
# （NBODY_FAST_MATH），不应该借着"CPU 基线要向量化"顺带引入。
function(nbody_apply_native_flags target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:/O2 /arch:AVX2>>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3 -march=native>>)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# CPU 基线第 1 档：单线程标量、关闭向量化。
#
# 必须是独立的库/编译单元：编译选项是文件级的，若和 kScalarO3 同处一个 TU，
# 两档会编成完全相同的代码，"向量化带来多少提升"就无从测量，
# 报告里的三档对比会退化成同一个数字。
# ---------------------------------------------------------------------------
add_library(nbody_cpu_naive STATIC src/cpu_naive.cpp)
target_include_directories(nbody_cpu_naive PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NBODY_FP64)
  target_compile_definitions(nbody_cpu_naive PRIVATE NBODY_FP64)
endif()
if(MSVC)
  target_compile_definitions(nbody_cpu_naive PRIVATE _CRT_SECURE_NO_WARNINGS)
  # /O2 保留标量优化，但显式禁用自动向量化（/Qvec-）。
  # 不用未文档化的 /d2Qvec-：在部分 cl.exe 版本上会被当作无法识别的
  # /d2 选项而静默忽略，向量化实际并未关闭，三档对比就失去意义。
  target_compile_options(nbody_cpu_naive PRIVATE /utf-8 /O2 /Qvec- /fp:precise)
else()
  # gcc 的 SLP（同基本块内）向量化开关拼法是 -fno-tree-slp-vectorize；
  # -fno-slp-vectorize 是 clang 的写法，gcc 不认识，直接编译失败。
  # -ftree-vectorize 是循环向量化，-ftree-slp-vectorize 是 SLP 向量化，
  # 两者独立控制，必须都关掉才能真正得到"未向量化"的基线。
  target_compile_options(nbody_cpu_naive PRIVATE
    -O2 -fno-tree-vectorize -fno-tree-slp-vectorize)
endif()

# ---------------------------------------------------------------------------
# 主模拟程序
# ---------------------------------------------------------------------------
add_executable(nbody_sim
  src/main.cu
  src/kernels.cu
  src/snapshot_stream.cu
  src/diagnostics_gpu.cu
  src/cuda_utils.cu
  src/sim_params.cpp
  src/trajectory_io.cpp
  src/diagnostics.cpp
  src/cpu_reference.cpp
)
nbody_apply_host_flags(nbody_sim)
nbody_apply_native_flags(nbody_sim)
target_link_libraries(nbody_sim PRIVATE nbody_cpu_naive)

target_compile_options(nbody_sim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  # 生成行号信息，ncu / nsys 才能把指标映射回源码行。
  $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-lineinfo>>
)
# .cu 文件的宿主编译同样需要 /utf-8，必须经 -Xcompiler 转发给 cl.exe。
if(MSVC)
  target_compile_options(nbody_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(NBODY_FAST_MATH)
  target_compile_options(nbody_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()

# nvcc 需要把 OpenMP 标志通过 -Xcompiler 转给宿主编译器，
# 直接把 OpenMP::OpenMP_CXX 链上去在 .cu 文件上会失败，所以手动处理。
if(OpenMP_CXX_FOUND)
  target_link_libraries(nbody_sim PRIVATE OpenMP::OpenMP_CXX)
  if(MSVC)
    target_compile_options(nbody_sim PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
  else()
    target_compile_options(nbody_sim PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif()
endif()

set_target_properties(nbody_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# 初值生成器（纯 CPU）
# ---------------------------------------------------------------------------
add_executable(nbody_ic src/ic_generator.cpp)
nbody_apply_host_flags(nbody_ic)
nbody_apply_native_flags(nbody_ic)

# ---------------------------------------------------------------------------
# 说明：本工程不包含单测目标（不启用 CTest，也没有 tests/ 目录）。
# 正确性门禁全部落在 scripts/validate.sh（8 段 16 项，PASS/FAIL 表），
# 包括对二进制粒子主序布局的字节级交叉判定（段 1）——
# 它只用 nbody_ic + nbody_sim + od/dd/cmp，不需要任何测试框架。
# ---------------------------------------------------------------------------
