CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 75
BUILD_DIR ?= build
WITH_HW_PATH ?= 0
WITH_HADAMARD ?= 0
CUBLAS_HOME ?=

# 默认按 T4 的 SM75 编译；调用者可通过 CUDA_ARCH 覆盖为实际 GPU 架构。
# WITH_HW_PATH=1 打开硬件/库对照路径（加分项），默认为 0 时产物与之前一致。
# WITH_HADAMARD=1 打开 Hadamard 旋转融合（链接 ../../03_hadamard_tc/yangyeyuan），默认为 0。
.PHONY: all configure test clean

all: configure
	cmake --build $(BUILD_DIR) -j

configure:
	# 把 nvcc 和架构显式传给 CMake，便于在多 CUDA 安装环境中复现构建。
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DWITH_HW_PATH=$(WITH_HW_PATH) -DWITH_HADAMARD=$(WITH_HADAMARD) \
		$(if $(CUBLAS_HOME),-DCUBLAS_ROOT=$(CUBLAS_HOME),)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake -E remove_directory $(BUILD_DIR)
