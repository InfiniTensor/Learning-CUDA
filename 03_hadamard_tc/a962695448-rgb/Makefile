CUDA_HOME ?= $(HOME)/.local/opt/cuda-12.8
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= 89
CXX ?= g++
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=sm_$(ARCH)

.PHONY: all self-test benchmark cpu-test clean FORCE
all: build/hadamard

build/config.txt: FORCE
	@mkdir -p build
	@printf '%s\n' '$(NVCC) $(NVCCFLAGS)' > build/config.tmp
	@cmp -s build/config.tmp $@ || cp build/config.tmp $@
	@rm -f build/config.tmp

build/hadamard: src/main.cu include/kernels.cuh include/contiguous256.cuh include/packed_rows.cuh include/row_policy.hpp include/reference.hpp build/config.txt
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -Iinclude src/main.cu -o $@

self-test: build/hadamard
	./build/hadamard --self-test

benchmark: build/hadamard
	./build/hadamard --benchmark --csv results/benchmark.csv

cpu-test: tests/cpu_reference_test.cpp include/reference.hpp tests/row_policy_test.cpp include/row_policy.hpp
	mkdir -p build
	$(CXX) -O3 -std=c++17 -Iinclude tests/cpu_reference_test.cpp -o build/cpu_reference_test
	./build/cpu_reference_test
	$(CXX) -O3 -std=c++17 -Iinclude tests/row_policy_test.cpp -o build/row_policy_test
	./build/row_policy_test

clean:
	rm -f build/hadamard build/cpu_reference_test build/row_policy_test build/config.txt build/config.tmp
