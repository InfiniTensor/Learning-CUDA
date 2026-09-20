CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
# 默认生成 sm_80，也可以通过 CUDA_ARCH=120 等参数覆盖。
CUDA_ARCH ?= 80
BUILD_DIR ?= build
CUBLAS_ROOT ?= $(CUDA_HOME)

NVCC_FLAGS := -std=c++17 -O3 --use_fast_math -lineinfo \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) \
	-Iinclude -I$(CUBLAS_ROOT)/include
LD_FLAGS := -L$(CUBLAS_ROOT)/lib64 -L$(CUBLAS_ROOT)/lib \
	-lcublas -lcudart

.PHONY: all test benchmark clean

# 默认目标只构建 benchmark 可执行文件。
all: $(BUILD_DIR)/hadamard_bench

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/hadamard.o: src/hadamard.cu include/hadamard.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cu include/hadamard.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/hadamard_bench: $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/main.o
	$(NVCC) $^ $(LD_FLAGS) -o $@

test: $(BUILD_DIR)/hadamard_bench
	./$(BUILD_DIR)/hadamard_bench --self-test

# benchmark 脚本会清空并重新生成 results/performance.csv。
benchmark: $(BUILD_DIR)/hadamard_bench
	./scripts/run_benchmarks.sh

clean:
	rm -f $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/main.o $(BUILD_DIR)/hadamard_bench
