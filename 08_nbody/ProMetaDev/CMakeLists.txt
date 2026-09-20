# ============================================================
# CudaNBodyGravitySim_2026summer —— CMake 编译配置
#   与 CudaDerivativesPricing_2026summer 项目 100% 隔离, 零共享代码
#   CUDA 架构: sm_75(Turing) / sm_86(Ampere) / sm_90(Hopper)
#             + compute_90 → PTX 供 Blackwell (sm_120) JIT
# ============================================================
cmake_minimum_required(VERSION 3.14)
project(CudaNBodyGravitySim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CUDA_STANDARD         14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 输出目录 (可选, 约定 build/ 下产生 nbody 可执行)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# -------- CUDA SM 架构选择策略: NBODY_SM_LIST (cmake 传参) > 自动 detect_sm.sh > 默认 86 --------
set(DETECTED_SM "")
if (DEFINED NBODY_SM_LIST AND NBODY_SM_LIST)
    set(DETECTED_SM "${NBODY_SM_LIST}")
    message(STATUS "[NBody] Using SM list from -DNBODY_SM_LIST = ${DETECTED_SM}")
else()
    execute_process(
        COMMAND bash -c "${CMAKE_SOURCE_DIR}/scripts/detect_sm.sh 2>/dev/null || echo '86'"
        OUTPUT_VARIABLE DETECTED_SM
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "[NBody] Auto detected SM = ${DETECTED_SM}")
endif()
if (NOT DETECTED_SM)
    set(DETECTED_SM "75,80,86,89,90")
endif()
message(STATUS "[NBody] Final SM tokens = ${DETECTED_SM}")

# -------- NVCC 架构生成 (健壮: 去除逗号/重复, 且保留 sm_75/86/90 三档显式兼容) --------
# 先清洗 DETECTED_SM: 取出所有纯数字 token, 再和标准 75/86/90 合并去重
string(REGEX MATCHALL "[0-9]+" _DET_SMS "${DETECTED_SM}")
list(APPEND _DET_SMS 75 86 90)
list(REMOVE_DUPLICATES _DET_SMS)
foreach(SM IN LISTS _DET_SMS)
    if (SM GREATER_EQUAL 50 AND SM LESS_EQUAL 999)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM},code=sm_${SM}")
    endif()
endforeach()
# PTX forward compat (Blackwell sm_120 JIT 通过 compute_90 PTX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=compute_90")

# -------- 编译警告 & 优化 --------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -Wextra")

# -------- include 路径 --------
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------- 可执行目标: nbody --------
add_executable(nbody
    src/main.cpp
    src/nbody_kernels.cu
    src/file_io.cpp
    src/analysis.cpp
)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nbody PRIVATE cudart)

message(STATUS "[NBody] Target = nbody, sources = main.cpp + nbody_kernels.cu + file_io.cpp + analysis.cpp")

# ============================================================
# Install (打包发布)
#   用法: cd build && cmake --install . --prefix ../dist
#   产物结构:
#     dist/bin/nbody
#     dist/scripts/*.py  *.sh
#     dist/data/         (样例粒子/参数)
#     dist/docs/         (report.md)
# ============================================================
include(GNUInstallDirs)
install(TARGETS nbody RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY scripts/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/nbody/scripts
        FILES_MATCHING PATTERN "*.py" PATTERN "*.sh"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

if (IS_DIRECTORY "${CMAKE_SOURCE_DIR}/data")
    install(DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/nbody/data)
endif()
if (EXISTS "${CMAKE_SOURCE_DIR}/report.md")
    install(FILES report.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
if (IS_DIRECTORY "${CMAKE_SOURCE_DIR}/docs")
    install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
if (EXISTS "${CMAKE_SOURCE_DIR}/README.md")
    install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
if (EXISTS "${CMAKE_SOURCE_DIR}/progress_report.csv")
    install(FILES progress_report.csv DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
