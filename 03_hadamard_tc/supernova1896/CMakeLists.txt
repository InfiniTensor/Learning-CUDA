cmake_minimum_required(VERSION 3.20)

option(HADAMARD_ENABLE_MACA "启用沐曦曦云 C500 的 MACA 后端" OFF)
option(HADAMARD_ENABLE_MUSA "启用摩尔线程 MTT S4000 的 MUSA 后端" OFF)

if(HADAMARD_ENABLE_MACA OR HADAMARD_ENABLE_MUSA)
  project(hadamard_transform_accel LANGUAGES CXX)
else()
  project(hadamard_transform_accel LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT HADAMARD_ENABLE_MACA AND NOT HADAMARD_ENABLE_MUSA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
  endif()

  add_library(hadamard_cuda STATIC
    src/hadamard_cuda.cu
  )
  target_include_directories(hadamard_cuda PUBLIC include)
  set_target_properties(hadamard_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  add_executable(hadamard_bench src/hadamard_bench.cu)
  target_link_libraries(hadamard_bench PRIVATE hadamard_cuda)

  add_executable(hadamard_tests tests/test_hadamard.cu)
  target_link_libraries(hadamard_tests PRIVATE hadamard_cuda)

  add_executable(hadamard_cpu_bench src/hadamard_cpu_bench.cpp)

  enable_testing()
  add_test(NAME correctness COMMAND hadamard_tests)
elseif(HADAMARD_ENABLE_MACA)
  # MACA 的 CMake 语言模块由 SDK 提供；这里显式补充 runtime_cu 和
  # symbol_cu，解决 CUDA 兼容 API（wcuda*）的宿主链接。
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "MACA SDK 路径")
  if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca")
  endif()
  set(ENV{MACA_PATH} "${MACA_PATH}")

  find_package(MetaX REQUIRED)
  set(ENV{CUCC_PATH} "${MetaX_CUBRIDGE_ROOT}")
  set(ENV{CUDA_PATH} "${MetaX_CUBRIDGE_ROOT}")
  list(APPEND CMAKE_MODULE_PATH "${MetaX_CMAKE_MODULE_DIR}")
  set(CMAKE_MACA_SOURCE_FILE_EXTENSIONS maca)
  enable_language(MACA)
  # MACA 3.7.1.5 自带的 CMake 模板没有展开 <INCLUDES> 和
  # <LINK_LIBRARIES>，会导致 __macro_mxcc.h 或 wcuda* 链接失败。
  set(CMAKE_INCLUDE_FLAG_MACA "-I")
  set(CMAKE_MACA_COMPILE_OBJECT
    "<CMAKE_MACA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -xmaca -o <OBJECT> -c <SOURCE>"
  )
  set(CMAKE_MACA_LINK_EXECUTABLE
    "<CMAKE_MACA_COMPILER> <FLAGS> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> --maca-path=$ENV{MACA_PATH}"
  )
  if(NOT CMAKE_MACA_CREATE_STATIC_LIBRARY)
    # 当前 MACA SDK 未提供该变量，补充标准 ar/ranlib 静态库规则。
    set(CMAKE_MACA_CREATE_STATIC_LIBRARY
      "<CMAKE_AR> qc <TARGET> <OBJECTS>"
      "<CMAKE_RANLIB> <TARGET>"
    )
  endif()

  find_library(HADAMARD_RUNTIME_CU_LIBRARY
    NAMES runtime_cu
    PATHS "${MACA_PATH}/lib"
    REQUIRED
  )
  find_library(HADAMARD_SYMBOL_CU_LIBRARY
    NAMES symbol_cu
    PATHS "${MACA_PATH}/lib"
    REQUIRED
  )

  set(HADAMARD_MACA_COMMON_LIBRARIES
    ${MetaX_LIBRARIES}
    ${HADAMARD_RUNTIME_CU_LIBRARY}
    ${HADAMARD_SYMBOL_CU_LIBRARY}
  )

  add_library(hadamard_maca STATIC
    src/hadamard_maca.maca
  )
  set_source_files_properties(src/hadamard_maca.maca PROPERTIES LANGUAGE MACA)
  target_include_directories(hadamard_maca PUBLIC
    include
    "${MetaX_INCLUDE_DIRS}"
    "${MetaX_CUBRIDGE_ROOT}/include"
  )
  target_compile_definitions(hadamard_maca PRIVATE HADAMARD_MACA=1)
  target_compile_options(hadamard_maca PRIVATE -xmaca)
  target_link_libraries(hadamard_maca PUBLIC ${HADAMARD_MACA_COMMON_LIBRARIES})
  set_target_properties(hadamard_maca PROPERTIES
    LINKER_LANGUAGE MACA
    POSITION_INDEPENDENT_CODE ON
  )

  add_executable(hadamard_bench_maca src/hadamard_bench_maca.maca)
  set_source_files_properties(src/hadamard_bench_maca.maca PROPERTIES LANGUAGE MACA)
  target_include_directories(hadamard_bench_maca PRIVATE
    include
    "${MetaX_INCLUDE_DIRS}"
    "${MetaX_CUBRIDGE_ROOT}/include"
  )
  target_compile_definitions(hadamard_bench_maca PRIVATE HADAMARD_MACA=1)
  target_compile_options(hadamard_bench_maca PRIVATE -xmaca)
  target_link_libraries(hadamard_bench_maca PRIVATE hadamard_maca)
  set_target_properties(hadamard_bench_maca PROPERTIES LINKER_LANGUAGE MACA)

  add_executable(hadamard_tests_maca tests/test_hadamard_maca.maca)
  set_source_files_properties(tests/test_hadamard_maca.maca PROPERTIES LANGUAGE MACA)
  target_include_directories(hadamard_tests_maca PRIVATE
    include
    "${MetaX_INCLUDE_DIRS}"
    "${MetaX_CUBRIDGE_ROOT}/include"
  )
  target_compile_definitions(hadamard_tests_maca PRIVATE HADAMARD_MACA=1)
  target_compile_options(hadamard_tests_maca PRIVATE -xmaca)
  target_link_libraries(hadamard_tests_maca PRIVATE hadamard_maca)
  set_target_properties(hadamard_tests_maca PROPERTIES LINKER_LANGUAGE MACA)

  enable_testing()
  add_test(NAME correctness_maca COMMAND hadamard_tests_maca)
else()
  # MUSA SDK 的 mcc 是独立编译器；用 custom command 保持 CMake 在
  # Windows/NVIDIA/MACA 环境中仍然可以正常配置。
  set(MUSA_HOME "$ENV{MUSA_HOME}" CACHE PATH "MUSA SDK 路径")
  if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
  endif()
  find_program(HADAMARD_MUSA_MCC
    NAMES mcc
    PATHS "${MUSA_HOME}/bin"
    NO_DEFAULT_PATH
  )
  if(NOT HADAMARD_MUSA_MCC)
    message(FATAL_ERROR "未找到 mcc，请设置 MUSA_HOME")
  endif()
  find_program(HADAMARD_MUSA_BASH NAMES bash REQUIRED)

  add_custom_target(hadamard_musa ALL
    COMMAND "${CMAKE_COMMAND}" -E env
      "MUSA_HOME=${MUSA_HOME}"
      "BUILD_DIR=${CMAKE_CURRENT_SOURCE_DIR}/build-musa"
      "${HADAMARD_MUSA_BASH}"
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_musa.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "使用 mcc 构建 MUSA S4000 后端"
    VERBATIM
  )
  enable_testing()
  add_test(
    NAME correctness_musa
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build-musa/hadamard_tests_musa"
  )
endif()
