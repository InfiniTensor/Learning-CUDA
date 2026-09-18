cmake_minimum_required(VERSION 3.16)
project(cpu_nvfp4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpu_nvfp4 main.cpp)
target_compile_options(cpu_nvfp4 PRIVATE -Wall -Wextra -Wpedantic -O3)
