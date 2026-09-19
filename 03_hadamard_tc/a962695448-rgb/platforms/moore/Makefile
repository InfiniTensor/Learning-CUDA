MUSA_ROOT ?= /usr/local/musa
MCC ?= $(MUSA_ROOT)/bin/mcc
ARCH ?= mp_22
SHUFFLE32 ?= 0
BUILD_DIR ?= build/$(ARCH)/shuffle$(SHUFFLE32)
CXXFLAGS := -std=c++17 -O2 -fno-fast-math -ffp-contract=off --offload-arch=$(ARCH)
CPPFLAGS := -I../../include
LDLIBS := -L$(MUSA_ROOT)/lib -lmusart
ifeq ($(SHUFFLE32),1)
CPPFLAGS += -DHADAMARD_MOORE_SHUFFLE32
endif

.PHONY: all probe
all: $(BUILD_DIR)/validate_and_benchmark

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/validate_and_benchmark: hadamard_api.mu hadamard_api.h exact_int4.hpp validate_and_benchmark.mu ../../include/reference.hpp | $(BUILD_DIR)
	$(MCC) $(CXXFLAGS) $(CPPFLAGS) hadamard_api.mu validate_and_benchmark.mu $(LDLIBS) -o $@

$(BUILD_DIR)/probe: probe.mu | $(BUILD_DIR)
	$(MCC) $(CXXFLAGS) probe.mu $(LDLIBS) -o $@

probe: $(BUILD_DIR)/probe
	LD_LIBRARY_PATH=$(MUSA_ROOT)/lib:$(LD_LIBRARY_PATH) $(BUILD_DIR)/probe
