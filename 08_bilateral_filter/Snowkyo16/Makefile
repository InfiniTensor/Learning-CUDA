# *********************************************************************
# 双边滤波 CUDA 项目 - Makefile
# Targets:
#   make build                          : 仅编译
#   make run                            : 运行（默认 MODE=all）
#   make run MODE=v0                    : 只跑 CPU 版本
#   make run MODE=v1                    : 只跑 GPU V1
#   make run INPUT=test_images/xxx.png  : 指定输入图片
#   make clean                          : 清理临时文件
#
# GPU 选择（多卡环境下指定空闲 GPU）:
#   CUDA_VISIBLE_DEVICES=2 make run
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM        ?= nvidia
CFLAGS          := -std=c++17 -O2
INCLUDE         := -I include

ifeq ($(PLATFORM),nvidia)
    CC              := /usr/local/cuda/bin/nvcc
    CFLAGS          += -arch=sm_80 -DPLATFORM_NVIDIA
else
    $(error 仅支持 nvidia 平台)
endif

# 目录定义
BUILD_DIR       := build
OUTPUT_DIR      := output

# 可执行文件
TARGET          := $(BUILD_DIR)/bilateral_filter

# 目标文件：主程序 + CPU代码 + GPU kernel + 工具
MAIN_OBJ        := $(BUILD_DIR)/main.o
CU_OBJS 		:= $(BUILD_DIR)/kernels.o
CPP_OBJS   		:= $(BUILD_DIR)/bilateral_cpu.o $(BUILD_DIR)/image_io.o $(BUILD_DIR)/params.o $(BUILD_DIR)/benchmark.o 
ALL_OBJS 		:= $(MAIN_OBJ) $(CU_OBJS) $(CPP_OBJS)

# 用户可指定输入图像和运行版本
INPUT           ?=
MODE  			?= all

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: all build run clean

# 默认目标：编译 + 运行
all: build run

# 编译目标
build: $(TARGET)

# 运行目标
run: $(TARGET)
	@mkdir -p $(OUTPUT_DIR)/images
	@echo "=== 运行双边滤波 ($(MODE)) ==="
	@if [ -n "$(INPUT)" ]; then \
	    IMG="$(INPUT)"; \
	else \
	    IMG=$$(ls test_images/*.png test_images/*.jpg 2>/dev/null | head -1); \
	fi; \
	if [ -z "$$IMG" ]; then \
	    echo "错误: test_images/ 下没有找到 png/jpg 图片"; \
	    echo "用法: make run INPUT=test_images/xxx.png"; \
	    exit 1; \
	fi; \
	echo "输入图像: $$IMG"; \
	./$(TARGET) $$IMG params.txt $(OUTPUT_DIR) $(MODE)

# 清理目标
clean:
	@echo "=== 清理临时文件 ==="
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)

# -------------------------------
# Dependency Rules
# -------------------------------

# 链接
$(TARGET): $(ALL_OBJS)
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) -o $@ $^

# 编译.cu 文件 (main.cu, kernels.cu)
$(BUILD_DIR)/%.o: src/%.cu
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# 编译 .cpp 文件
$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
