MUXI_HOME ?= /opt/maca-3.5.3
MXCC ?= $(MUXI_HOME)/mxgpu_llvm/bin/mxcc
BUILD_DIR ?= build
MUXI_ARCH ?=
WARPS ?= 4
FUSED_WARPS ?= $(WARPS)
QUANT_VECTOR ?= 0
MUXI_EXTRA_FLAGS ?=

COMMON_FLAGS := -O3 -std=c++17 -I$(CURDIR)/src \
				-I$(MUXI_HOME)/include \
				-I$(MUXI_HOME)/tools/cu-bridge/include \
                -DHADAMARD_WARPS_PER_BLOCK=$(WARPS) \
                -DHADAMARD_FUSED_WARPS_PER_BLOCK=$(FUSED_WARPS) \
                -DHADAMARD_QUANT_VECTOR_STORE=$(QUANT_VECTOR) \
                $(MUXI_EXTRA_FLAGS)
LINK_FLAGS := -L$(MUXI_HOME)/lib -Wl,-rpath,$(MUXI_HOME)/lib -lruntime_cu
ifneq ($(strip $(MUXI_ARCH)),)
COMMON_FLAGS += --cuda-gpu-arch=$(MUXI_ARCH)
endif

BIN := $(BUILD_DIR)/hadamard_bench
ADV_BIN := $(BUILD_DIR)/hadamard_advanced_bench
OBJS := $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/hadamard_warp.o $(BUILD_DIR)/main.o
ADV_OBJS := $(BUILD_DIR)/hadamard.o $(BUILD_DIR)/hadamard_warp.o $(BUILD_DIR)/advanced_bench.o

.PHONY: all clean smoke check-toolchain

all: $(BIN) $(ADV_BIN)

check-toolchain:
	@test -x "$(MXCC)" || (echo "沐曦编译器不存在: $(MXCC). 请设置 MUXI_HOME 或 MXCC。"; exit 1)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR) check-toolchain
	$(MXCC) $(COMMON_FLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(MXCC) $(COMMON_FLAGS) $^ $(LINK_FLAGS) -o $@

$(ADV_BIN): $(ADV_OBJS)
	$(MXCC) $(COMMON_FLAGS) $^ $(LINK_FLAGS) -o $@

smoke: $(BIN)
	$(BIN) --batch 1 --seq 8 --heads 2 --head_dim 64 --dtype fp16 --normalize true --warmup 1 --iters 3 --check true

clean:
	rm -rf $(BUILD_DIR)
