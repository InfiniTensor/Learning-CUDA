# Makefile — 不依赖 CMake 的直接构建方式（本机无 cmake 时使用）
#
# 用法：
#   make                 # 构建 ./hadamard_bench
#   make CUDA_ARCH=86    # 指定目标架构（默认 86 = RTX 3060 Laptop/Ampere）
#   make clean
#   make smoke           # 构建并跑一个小的正确性检查
#
# 等价的 CMake 构建方式见 CMakeLists.txt：
#   mkdir build && cd build && cmake .. && make -j

NVCC      ?= nvcc
CUDA_ARCH ?= 86
OUT       ?= build
WARPS     ?= 4
FUSED_WARPS ?= $(WARPS)
QUANT_VECTOR ?= 0

NVCCFLAGS := -O3 -std=c++17 \
             -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
             -Iinclude \
             -DHADAMARD_WARPS_PER_BLOCK=$(WARPS) \
             -DHADAMARD_FUSED_WARPS_PER_BLOCK=$(FUSED_WARPS) \
             -DHADAMARD_QUANT_VECTOR_STORE=$(QUANT_VECTOR)

BIN := $(OUT)/hadamard_bench
ADV_BIN := $(OUT)/hadamard_advanced_bench
OBJS := $(OUT)/hadamard.o $(OUT)/hadamard_warp.o $(OUT)/main.o
ADV_OBJS := $(OUT)/hadamard.o $(OUT)/hadamard_warp.o $(OUT)/advanced_bench.o

.PHONY: all clean smoke run_sweep

all: $(BIN) $(ADV_BIN)

$(OUT):
	mkdir -p $(OUT)

$(OUT)/hadamard.o: src/hadamard.cu include/hadamard.cuh include/cuda_check.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/hadamard_warp.o: src/hadamard_warp.cu include/hadamard.cuh include/cuda_check.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/main.o: src/main.cu src/reference_cpu.h include/hadamard.cuh include/cuda_check.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/advanced_bench.o: src/advanced_bench.cu include/hadamard.cuh include/cuda_check.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(ADV_BIN): $(ADV_OBJS)
	$(NVCC) $(NVCCFLAGS) $(ADV_OBJS) -o $@

smoke: $(BIN) $(ADV_BIN)
	$(BIN) --batch 1 --seq 64 --heads 4 --head_dim 128 --dtype fp16 \
	       --normalize true --warmup 2 --iters 5 --check true
	$(ADV_BIN) --batch 1 --seq 32 --heads 2 --head_dim 128 --dtype fp16 \
	       --normalize true --warmup 2 --iters 5

run_sweep: $(BIN)
	bash scripts/run_tests.sh

clean:
	rm -rf $(OUT)
