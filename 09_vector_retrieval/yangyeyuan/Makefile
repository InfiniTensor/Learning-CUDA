CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120
BUILD_DIR ?= build

.PHONY: all configure build test clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory $(BUILD_DIR)
