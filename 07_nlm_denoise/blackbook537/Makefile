# NLM denoise: make build / run / validate / test / bench / clean
.DEFAULT_GOAL := build
SHELL := /bin/bash
.DELETE_ON_ERROR:

PLATFORM    ?= nvidia
WITH_OPENCV ?= 0
FAST_EXP    ?= 0

# 可选覆盖变量（默认即训练营服务器环境，无需设置）：
#   NVCC     nvcc 路径（默认取 PATH 中的 nvcc）
#   CXX_HOST nvcc 的主机编译器（本地 gcc 版本过新时指定，如 conda gcc13）
#   CUDA_RT  非系统位置的 cudart 运行库目录（需含 include/ 与 lib/）
#   ARCH     目标 GPU 架构（如 sm_80/sm_86/sm_90；留空用 nvcc 默认全架构，
#            会产生 pre-sm_75 弃用警告并增加编译时间）
NVCC        ?= nvcc
CXX_HOST    ?=
CUDA_RT     ?=
ARCH        ?=
COREX_HOME  ?= /usr/local/corex
COREX_CXX   ?= $(COREX_HOME)/bin/clang++
MUSA_HOME   ?= /usr/local/musa
MCC         ?= $(MUSA_HOME)/bin/mcc
MACA_HOME   ?= /opt/maca
MACA_CUDA   ?= $(MACA_HOME)/tools/cu-bridge
MXCC        ?= $(MACA_HOME)/mxgpu_llvm/bin/mxcc

SUFFIX          := cu
PLATFORM_DEFINE := -DPLATFORM_NVIDIA
CFLAGS          := -std=c++17 -O3
EXTRA_LIBS      :=
COMP_ONLY_FLAGS :=

ifeq ($(PLATFORM),nvidia)
    CC              := $(NVCC)
    PLATFORM_DEFINE := -DPLATFORM_NVIDIA
    ifneq ($(CXX_HOST),)
        CFLAGS += -ccbin $(CXX_HOST)
    endif
    ifneq ($(CUDA_RT),)
        CFLAGS     += -I$(CUDA_RT)/include
        EXTRA_LIBS += -L$(CUDA_RT)/lib -lcudart_static -lpthread -ldl -lrt
    endif
    ifneq ($(ARCH),)
        CFLAGS += -arch=$(ARCH)
    endif
else ifeq ($(PLATFORM),iluvatar)
    CC              := $(COREX_CXX)
    CFLAGS          := -std=c++17 -O3 -I$(COREX_HOME)/include -fPIC
    COMP_ONLY_FLAGS := -x ivcore --cuda-path=$(COREX_HOME)
    PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
    EXTRA_LIBS      := -L$(COREX_HOME)/lib64 -Wl,-rpath,$(COREX_HOME)/lib64 -lcudart
else ifeq ($(PLATFORM),metax)
    CC              := $(MXCC)
    CFLAGS          := -std=c++17 -O3 -I$(MACA_CUDA)/include
    SUFFIX          := maca
    PLATFORM_DEFINE := -DPLATFORM_METAX
    # MACA 3.x 的 CUDA 兼容头位于 cu-bridge；将设备对象单独链接时还需
    # 显式链接 wcuda 符号实现。rpath 使产物无需额外设置 LD_LIBRARY_PATH。
    EXTRA_LIBS      := -L$(MACA_HOME)/lib -Wl,-rpath,$(MACA_HOME)/lib \
                       -lruntime_cu -lsymbol_cu
else ifeq ($(PLATFORM),moore)
    CC              := $(MCC)
    CFLAGS          := -std=c++11 -O3
    SUFFIX          := mu
    PLATFORM_DEFINE := -DPLATFORM_MOORE
    # rpath 让构建产物在未手工设置 LD_LIBRARY_PATH 时也能找到 libmusart；
    # 避免绑定发行版特定的 GCC 目录，mcc 会自行选择兼容的 Host 工具链。
    EXTRA_LIBS      := -I$(MUSA_HOME)/include -L$(MUSA_HOME)/lib \
                       -Wl,-rpath,$(MUSA_HOME)/lib -lmusart
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, metax, moore))
endif

ifeq ($(FAST_EXP),1)
    PLATFORM_DEFINE += -DNLM_FAST_EXP
endif

ifeq ($(WITH_OPENCV),1)
    PLATFORM_DEFINE += -DWITH_OPENCV
    CFLAGS    += $(shell pkg-config --cflags opencv4)
    EXTRA_LIBS += $(shell pkg-config --libs opencv4)
endif


TARGET          := build/nlm_denoise
PLATFORM_TARGET := build/$(PLATFORM)/nlm_denoise
OBJ_DIR         := build/obj/$(PLATFORM)
INCLUDES        := -Iinclude
HOST_SRCS       := $(wildcard src/*.cpp)
HOST_OBJS       := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(HOST_SRCS))
KER_SRC         := src/kernels.$(SUFFIX)
KER_OBJ         := $(OBJ_DIR)/kernels.o
OBJS            := $(HOST_OBJS) $(KER_OBJ)
HEADERS         := $(wildcard include/*.h)

INPUT       ?= test_images/noisy_256x256_3ch_sigma25.png
OUTPUT      ?= output/images/denoised.png
PARAMS      ?= params.txt
MODE        ?= $(if $(filter metax,$(PLATFORM)),1,2)
RUN_LOG     ?= output/logs/nlm_perf.csv
SIZES       ?= 1920x1080,3840x2160
CHANNELS    ?= 1,3
PARAM_SETS  ?= all
WARMUP      ?= 3
REPEAT      ?= 10
BENCH_LOG   ?= output/results/benchmark.csv

.PHONY: all build run validate test bench clean FORCE
all: build
build: $(PLATFORM_TARGET)

# Keep the documented CLI path while all recipes use their own platform binary.
$(TARGET): $(PLATFORM_TARGET) FORCE
	@cp -f $(PLATFORM_TARGET) $(TARGET)
build: $(TARGET)

run: build
	$(PLATFORM_TARGET) run -i "$(INPUT)" -o "$(OUTPUT)" -p "$(PARAMS)" --kernel $(MODE) --log "$(RUN_LOG)"

validate: build
	$(PLATFORM_TARGET) validate -i "$(INPUT)" -o "$(OUTPUT)" -p "$(PARAMS)"

test: build
	$(PLATFORM_TARGET) units
	$(PLATFORM_TARGET) validate -i "$(INPUT)" -p "$(PARAMS)"

bench: build
	$(PLATFORM_TARGET) bench --sizes "$(SIZES)" --channels "$(CHANNELS)" --param-sets "$(PARAM_SETS)" --warmup $(WARMUP) --repeat $(REPEAT) --log "$(BENCH_LOG)"

clean:
	rm -rf build

# Track compiler and flag changes without touching unchanged objects.
$(OBJ_DIR)/build_flags.txt: FORCE
	@mkdir -p $(OBJ_DIR)
	@printf '%s\n' '$(CC) $(CFLAGS) $(COMP_ONLY_FLAGS) $(PLATFORM_DEFINE) $(INCLUDES) $(EXTRA_LIBS)' > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

$(PLATFORM_TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $(OBJS) $(EXTRA_LIBS)

$(OBJ_DIR)/%.o: src/%.cpp $(HEADERS) $(OBJ_DIR)/build_flags.txt
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) $(INCLUDES) -c $< -o $@

$(KER_OBJ): $(KER_SRC) src/kernels.cu $(HEADERS) $(OBJ_DIR)/build_flags.txt
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(COMP_ONLY_FLAGS) $(PLATFORM_DEFINE) $(INCLUDES) -c $(KER_SRC) -o $@
