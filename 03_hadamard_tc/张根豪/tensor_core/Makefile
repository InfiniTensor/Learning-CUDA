# Makefile — Tensor Core Hadamard 对照实验
#
# 用法：
#   make                  # 构建 ./build/hadamard_tc_bench
#   make CUDA_ARCH=89     # 指定架构（L40S=89, A100=80, H100/H200=90）
#   make smoke            # 小规模正确性冒烟
#   make clean
#
# 注意：BF16 WMMA 需要 SM >= 80，本模块不支持 SM 7.x。

NVCC      ?= nvcc
CUDA_ARCH ?= 80
OUT       ?= build
ROOT      ?= ..
WARPS     ?= 4
FUSED_WARPS ?= $(WARPS)
QUANT_VECTOR ?= 0
TC_WARPS  ?= 4
TC_VECTOR ?= 0
MMA_PRECISION ?= 0

NVCCFLAGS := -O3 -std=c++17 \
             -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
             -I$(ROOT)/include -Iinclude \
             -DHADAMARD_WARPS_PER_BLOCK=$(WARPS) \
             -DHADAMARD_FUSED_WARPS_PER_BLOCK=$(FUSED_WARPS) \
             -DHADAMARD_QUANT_VECTOR_STORE=$(QUANT_VECTOR) \
             -DHADAMARD_TC_WARPS_PER_BLOCK=$(TC_WARPS) \
             -DHADAMARD_TC_VECTOR_SMEM=$(TC_VECTOR) \
             -DHADAMARD_MMA_PRECISION=$(MMA_PRECISION)

BIN  := $(OUT)/hadamard_tc_bench
OBJS := $(OUT)/hadamard.o $(OUT)/hadamard_warp.o $(OUT)/hadamard_tc.o $(OUT)/tc_bench.o

.PHONY: all clean smoke sweep h200-tuned bridge mma-experiment

all: $(BIN)

# Opt-in preset, never silently substitute a numerically different TC backend.
h200-tuned:
	$(MAKE) CUDA_ARCH=90 OUT=$(OUT)/h200-tuned WARPS=0 TC_WARPS=4 TC_VECTOR=1 all

# Optional benchmark C ABI; not linked into the production executable.
bridge: $(OUT)/libhadamard.so
$(OUT)/libhadamard.so: $(ROOT)/src/hadamard.cu $(ROOT)/src/hadamard_warp.cu src/python_bridge.cu $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC $(filter %.cu,$^) -o $@

# Isolated register-MMA experiment (d=64/128/256 only), never default dispatch.
MMA_SOURCES := $(ROOT)/src/hadamard.cu $(ROOT)/src/hadamard_warp.cu src/hadamard_tc.cu src/hadamard_mma_experimental.cu
mma-experiment: $(OUT)/libhadamard_mma.so $(OUT)/mma_smoke
$(OUT)/libhadamard_mma.so: $(MMA_SOURCES) src/python_bridge.cu include/hadamard_tc.cuh $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC -DHADAMARD_BRIDGE_TC -DHADAMARD_BRIDGE_MMA $(filter %.cu,$^) -o $@
$(OUT)/mma_smoke: $(MMA_SOURCES) src/mma_smoke.cu include/hadamard_tc.cuh $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu,$^) -o $@

$(OUT):
	mkdir -p $(OUT)

$(OUT)/hadamard.o: $(ROOT)/src/hadamard.cu $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/hadamard_warp.o: $(ROOT)/src/hadamard_warp.cu $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/hadamard_tc.o: src/hadamard_tc.cu include/hadamard_tc.cuh $(ROOT)/include/hadamard.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT)/tc_bench.o: src/tc_bench.cu include/hadamard_tc.cuh | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

smoke: $(BIN)
	$(BIN) --batch 1 --seq 64 --heads 4 --head_dim 64  --dtype fp16 --warmup 2 --iters 5
	$(BIN) --batch 1 --seq 64 --heads 4 --head_dim 128 --dtype fp16 --warmup 2 --iters 5
	$(BIN) --batch 1 --seq 64 --heads 4 --head_dim 256 --dtype bf16 --warmup 2 --iters 5

sweep: $(BIN)
	bash scripts/run_tc.sh

clean:
	rm -rf $(OUT)
