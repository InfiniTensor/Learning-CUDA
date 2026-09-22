NVCC ?= /usr/local/cuda/bin/nvcc
FLAGS := -O3 -std=c++17 --ftz=false --fmad=false -lineinfo -Xcompiler -fPIC
ARCH := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_75,code=compute_75

all: libcodec.so

extras: all mxfp8_native_minimal.so

libcodec.so: codec.cu formats.cuh
	$(NVCC) $(FLAGS) $(ARCH) -shared $< -o $@

mxfp8_native_minimal.so: mxfp8_native_minimal.cu
	$(NVCC) $(FLAGS) $(ARCH) -shared $< -o $@
