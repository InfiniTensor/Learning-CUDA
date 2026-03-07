# *********************************************************************
# V0: CPU 基线
# Targets:
#   make                      : Build + run (default)
#   make build                : Only compile
#   make run                  : Run bilateral filter
#   make run INPUT=xxx.png    : Run with specific image
#   make clean                : Delete temporary files
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM        ?= nvidia
CFLAGS          := -std=c++17 -O2
INCLUDE         := -I include

ifeq ($(PLATFORM),nvidia)
    CC              := /usr/local/cuda/bin/nvcc
    CFLAGS          += -arch=sm_80
else
    $(error V0 仅支持 nvidia 平台，GPU kernel 将在 V1 引入)
endif

# 目录定义
BUILD_DIR       := build
OUTPUT_DIR      := output

# 可执行文件
TARGET          := $(BUILD_DIR)/bilateral_filter

MAIN_OBJ        := $(BUILD_DIR)/main.o
CPP_OBJS        := $(BUILD_DIR)/bilateral_cpu.o $(BUILD_DIR)/image_io.o $(BUILD_DIR)/params.o
ALL_OBJS        := $(MAIN_OBJ) $(CPP_OBJS)

# 用户可指定输入图像（默认自动扫描 test_images/ 下第一张）
INPUT           ?=

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
	@echo "=== 运行双边滤波 (CPU) ==="
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
	./$(TARGET) $$IMG params.txt $(OUTPUT_DIR)

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

# 编译主程序（.cu 文件）
$(BUILD_DIR)/main.o: src/main.cu
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# 编译 CPU 端 .cpp 文件
$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
