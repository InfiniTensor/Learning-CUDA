cmake_minimum_required(VERSION 3.24)
project(nbody_cuda VERSION 1.0.0 LANGUAGES CXX)
option(NBODY_ENABLE_CUDA "Build the NVIDIA CUDA backend" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(nbody src/main.cpp src/io.cpp src/cpu.cpp)
target_include_directories(nbody PRIVATE include)
# Also track the project header explicitly: localized MSVC /showIncludes output
# can be decoded differently by Windows CMake/Ninja installations.
set_source_files_properties(src/main.cpp src/io.cpp src/cpu.cpp PROPERTIES
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/nbody.hpp")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
  target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2;/W4;/utf-8;/EHsc>)
else()
  target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>)
endif()
if(NBODY_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(nbody PRIVATE src/cuda_backend.cu)
  target_compile_definitions(nbody PRIVATE NBODY_HAS_CUDA=1)
  set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Static)
  target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
  target_link_libraries(nbody PRIVATE CUDA::cudart_static)
endif()
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME nbody_acceptance COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nbody.py" --exe $<TARGET_FILE:nbody>)
    add_test(NAME trajectory_reader COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trajectory_reader.py")
    add_test(NAME package_guard COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_package.py")
    add_test(NAME profile_capture_guard COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_verify_nsys.py")
  endif()
endif()
