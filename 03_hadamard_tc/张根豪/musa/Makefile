MUSA_HOME ?= /usr/local/musa-5.1.0
BUILD_DIR ?= build

.PHONY: all configure clean smoke

all: configure
	cmake --build $(BUILD_DIR) -j2

configure:
	PATH=$(MUSA_HOME)/bin:$$PATH LD_LIBRARY_PATH=$(MUSA_HOME)/lib:$$LD_LIBRARY_PATH \
	cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_MUSA_COMPILER=$(MUSA_HOME)/bin/mcc

smoke: all
	LD_LIBRARY_PATH=$(MUSA_HOME)/lib:$$LD_LIBRARY_PATH \
	./$(BUILD_DIR)/hadamard_bench --batch 1 --seq 8 --heads 2 \
		--head_dim 64 --dtype fp16 --normalize true --warmup 1 --iters 3 --check true

clean:
	rm -rf $(BUILD_DIR)