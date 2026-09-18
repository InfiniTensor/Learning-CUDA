COREX_HOME ?= /usr/local/corex-4.4.0
COREX_CXX ?= $(COREX_HOME)/bin/clang++
COREX_EXTRA_FLAGS ?=
BUILD_DIR ?= build
WARPS ?= 4
FUSED_WARPS ?= $(WARPS)
QUANT_VECTOR ?= 0

COMMON_FLAGS := -O3 -std=c++17 \
                -I$(CURDIR)/../include \
				-DTIANSHU_COREX \
                -DHADAMARD_WARPS_PER_BLOCK=$(WARPS) \
                -DHADAMARD_FUSED_WARPS_PER_BLOCK=$(FUSED_WARPS) \
                -DHADAMARD_QUANT_VECTOR_STORE=$(QUANT_VECTOR) \
                $(COREX_EXTRA_FLAGS)
DEVICE_FLAGS := -x ivcore --cuda-path=$(COREX_HOME)

BIN := $(BUILD_DIR)/hadamard_bench
ADV_BIN := $(BUILD_DIR)/hadamard_advanced_bench
OBJS := $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/hadamard_warp.o $(BUILD_DIR)/main.o
ADV_OBJS := $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/hadamard_warp.o $(BUILD_DIR)/advanced_bench.o

.PHONY: all clean smoke check-toolchain

all: $(BIN) $(ADV_BIN)

check-toolchain:
	@test -x "$(COREX_CXX)" || (echo "CoreX 编译器不存在: $(COREX_CXX). 请设置 COREX_CXX。"; exit 1)
	@"$(COREX_CXX)" --help 2>&1 | grep -q -- '--cuda-compile-host-device' || (echo "COREX_CXX 不是支持 CUDA/ivcore 的 CoreX clang++: $(COREX_CXX)."; exit 1)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: ../src/%.cu | $(BUILD_DIR) check-toolchain
	$(COREX_CXX) $(DEVICE_FLAGS) $(COMMON_FLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(COREX_CXX) $^ -L$(COREX_HOME)/lib64 -Wl,-rpath,$(COREX_HOME)/lib64 -lcudart -o $@

$(ADV_BIN): $(ADV_OBJS)
	$(COREX_CXX) $^ -L$(COREX_HOME)/lib64 -Wl,-rpath,$(COREX_HOME)/lib64 -lcudart -o $@

smoke: $(BIN)
	COREX_LD_LIBRARY_PATH=$(COREX_HOME)/lib64 LD_LIBRARY_PATH=$(COREX_HOME)/lib64:$$LD_LIBRARY_PATH \
	$(BIN) --batch 1 --seq 8 --heads 2 --head_dim 64 \
		--dtype fp16 --normalize true --warmup 1 --iters 3 --check true

clean:
	rm -rf $(BUILD_DIR)
