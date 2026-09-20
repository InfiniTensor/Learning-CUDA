# 选题二：MXFP8 / NVFP4 低精度模拟与反量化（CUDA 实现）
#
# 目录结构：
#   include/ + src/   库（无 main、无命令行、不产出可执行文件）
#   test/             测试程序（唯一产出可执行文件的地方）
#
# 常用目标:
#   make              # 构建测试程序（库作为目标文件一并编译，不单独产出可执行文件）
#   make test         # ★ 一行命令跑全套测试（正确性 + 性能 + nsys/ncu）
#   make test-fast    # 同上，但跳过 nsys / ncu（只出正确性与性能数据）
#   make clean
#
# -arch 自动按 GPU 计算能力探测，无需手动指定；如需强制: make ARCH="-arch=sm_86"
#
# 注意题目约束：不得依赖 Ampere 以上架构特性，本代码只用了 sm_35
# 即有的原语（grid-stride / warp shuffle / atomicMax），换架构只是重编译。

# CUDA 工具链：优先用 /usr/local/cuda/bin（很多机器没把 CUDA 加进 PATH），
# 找不到再回落 PATH 里的 nvcc。可用 make NVCC=... 覆盖。
NVCC ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc) nvcc)

# 编译目标：未显式指定时按 GPU 计算能力自动选择（探测不到则回落 sm_75 = T4）
NVIDIA_SMI ?= $(firstword $(wildcard /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi) nvidia-smi)
ARCH ?= $(shell cc=$$($(NVIDIA_SMI) --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d ' .'); if [ -n "$$cc" ]; then echo "-arch=sm_$$cc"; else echo "-arch=sm_75"; fi)

HDRS  = include/quant_format.h include/quant_io.h include/quant_gpu.h include/quant_tables.h
FLAGS = $(ARCH) -O3 -std=c++17 -Xcompiler -Wall,-Wextra -Iinclude

.PHONY: all test test-fast clean

all: test/test_all

# ---- 库：只编译成目标文件，不链接成可执行文件 ----
build/quant_io.o: src/quant_io.cu $(HDRS)
	@mkdir -p build
	$(NVCC) $(FLAGS) -c -o $@ src/quant_io.cu

build/quant_gpu.o: src/quant_gpu.cu $(HDRS)
	@mkdir -p build
	$(NVCC) $(FLAGS) -c -o $@ src/quant_gpu.cu

# ---- 测试程序：本工程唯一产出可执行文件的地方 ----
test/test_all: test/test_all.cpp test/anchors.h test/gen_input.h $(HDRS) build/quant_io.o build/quant_gpu.o
	$(NVCC) $(FLAGS) -o $@ test/test_all.cpp build/quant_io.o build/quant_gpu.o

# 一行命令跑全套。测试主体是 C++（test/test_all）；
# shell 只负责包装 nsys / ncu 这两个外部分析工具。
test: test/test_all
	./test/test_all
	bash test/run.sh

# 同上，但不跑 nsys / ncu（快）
test-fast: test/test_all
	./test/test_all

clean:
	rm -f test/test_all
	rm -rf build test/out
