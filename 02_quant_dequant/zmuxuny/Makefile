ifeq ($(PLATFORM),ascend)
include ascend/Makefile
else
PLATFORM ?= nvidia
MACA_PATH ?= /opt/maca
COREX_PATH ?= /usr/local/corex
IVCORE_ARCH ?= ivcore11
MUSA_PATH ?= /usr/local/musa
MUSA_ARCH ?= mp_22
NVCC ?= nvcc
HOSTCXX ?= $(shell command -v g++-10 2>/dev/null || command -v g++)
ARCH ?= 75
CXXFLAGS := -ccbin $(HOSTCXX) -std=c++17 -O3 -lineinfo --fmad=false -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
ifeq ($(PLATFORM),metax)
export MACA_PATH
export CUDA_PATH := $(MACA_PATH)/tools/cu-bridge
NVCC := $(MACA_PATH)/tools/cu-bridge/bin/cucc
CXXFLAGS := -std=c++17 -O3 --fmad=false -I$(MACA_PATH)/tools/cu-bridge/include
BUILD_DIR := build/metax
RUN_ENV := LD_LIBRARY_PATH=$(MACA_PATH)/lib:$${LD_LIBRARY_PATH:-}
TEST_OUTPUT := results/c500/correctness_quick.json
else ifeq ($(PLATFORM),iluvatar)
# CoreX clang targets the device ISA directly; its bundled nvcc is not used.
NVCC := $(COREX_PATH)/bin/clang++
CXXFLAGS := -x ivcore --cuda-gpu-arch=$(IVCORE_ARCH) -std=c++17 -O3 -ffp-contract=off
LDLIBS := -L$(COREX_PATH)/lib64 -lcudart
BUILD_DIR := build/iluvatar
RUN_ENV := LD_LIBRARY_PATH=$(COREX_PATH)/lib64:$${LD_LIBRARY_PATH:-}
TEST_OUTPUT := results/iluvatar/correctness.json
else ifeq ($(PLATFORM),musa)
# Build original sources with an explicit runtime adapter; no generated source copy.
NVCC := $(MUSA_PATH)/bin/mcc
CXXFLAGS := -x musa --offload-arch=$(MUSA_ARCH) -std=c++17 -O3 -ffp-contract=off --prec-div=true -fno-musa-approx-transcendentals -fdenormal-fp-math=ieee
LDLIBS := -L$(MUSA_PATH)/lib -lmusart
BUILD_DIR := build/musa
RUN_ENV := MUSA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$(MUSA_PATH)/lib:$${LD_LIBRARY_PATH:-}
TEST_OUTPUT := results/musa/correctness.json
else ifeq ($(PLATFORM),nvidia)
BUILD_DIR := build
TEST_OUTPUT := results/correctness_quick.json
else
$(error Unsupported PLATFORM '$(PLATFORM)' (expected nvidia, metax, iluvatar or musa))
endif

.PHONY: all test numeric-test clean native
all: $(BUILD_DIR)/quantize
$(BUILD_DIR)/quantize: src/main.cu $(wildcard include/*) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDLIBS)
test: all numeric-test
	mkdir -p $(dir $(TEST_OUTPUT))
	$(RUN_ENV) python3 tests/validate.py --binary $(BUILD_DIR)/quantize --output $(TEST_OUTPUT) --quick
clean:
	rm -rf $(BUILD_DIR)

# Explicit bonus target; the default binary always uses software FP8 encoding.
ifneq ($(PLATFORM),nvidia)
native:
	$(error The native target uses NVIDIA FP8 instructions; use the default target for this platform)
else
native: $(BUILD_DIR)/quantize_native
$(BUILD_DIR)/quantize_native: src/main.cu $(wildcard include/*) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -DLP_NATIVE_FP8=1 $< -o $@ $(LDLIBS)
endif

# Check lane-dependent RNG separately so launch-layout regressions are explicit.
$(BUILD_DIR)/validate_numeric: tests/validate_numeric.cu $(wildcard include/*) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDLIBS)
numeric-test: $(BUILD_DIR)/validate_numeric
	$(RUN_ENV) $(BUILD_DIR)/validate_numeric

# Candidate scans use the same compiler flags and numeric semantics as the app.
$(BUILD_DIR)/tune_%: tests/tune_%.cu $(wildcard include/*) $(wildcard src/*.cuh) Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDLIBS)

endif
