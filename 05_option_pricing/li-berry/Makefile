# Makefile - 期权定价 CUDA 项目
# .cpp 用 g++ 编译（STL/OpenMP），.cu 用 nvcc 编译，最终链接必须用 nvcc

NVCC := nvcc
CXX  := g++
ARCH := -arch=sm_89
STD  := -std=c++17

NVCCFLAGS := -O3 $(ARCH) $(STD) -lineinfo
CXXFLAGS  := -O3 $(STD) -Wall -Wextra

BUILD := build
SRC   := src
BIN   := $(BUILD)/option_pricer

# 共享头（任一变化触发相关目标重编译）
SHARED_H := $(SRC)/types.h $(SRC)/rng.h $(SRC)/payoff.h $(SRC)/sim_core.h \
            $(SRC)/engine_common.h $(SRC)/stats.h

OBJS := $(BUILD)/config.o $(BUILD)/analytic.o $(BUILD)/stats.o \
        $(BUILD)/cpu_engine.o $(BUILD)/output.o $(BUILD)/main.o $(BUILD)/gpu_engine.o

.PHONY: all test test-units e2e convergence bench profile format clean

all: $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/config.o: $(SRC)/config.cpp $(SRC)/config.h $(SRC)/types.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/analytic.o: $(SRC)/analytic.cpp $(SRC)/analytic.h $(SRC)/types.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/stats.o: $(SRC)/stats.cpp $(SHARED_H) $(SRC)/analytic.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cpu_engine.o: $(SRC)/cpu_engine.cpp $(SRC)/cpu_engine.h $(SHARED_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

$(BUILD)/output.o: $(SRC)/output.cpp $(SRC)/output.h $(SHARED_H) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/main.o: $(SRC)/main.cpp $(SHARED_H) $(SRC)/analytic.h $(SRC)/config.h \
                  $(SRC)/cpu_engine.h $(SRC)/gpu_engine.cuh $(SRC)/output.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/gpu_engine.o: $(SRC)/gpu_engine.cu $(SRC)/gpu_engine.cuh $(SHARED_H) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(NVCC) $(ARCH) -o $@ $^ -lcudart -lgomp

# 纯 CPU 单元测试（无 GPU 依赖：解析解/解析器/RNG/统计量）
$(BUILD)/test_units: tests/test_units.cpp $(BUILD)/config.o $(BUILD)/analytic.o $(BUILD)/stats.o | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(SRC) -o $@ tests/test_units.cpp $(BUILD)/config.o $(BUILD)/analytic.o $(BUILD)/stats.o

test: test-units e2e

test-units: $(BUILD)/test_units
	$(BUILD)/test_units

e2e: $(BIN)
	bash tests/run_e2e.sh

convergence: $(BIN)
	bash scripts/convergence.sh

bench: $(BIN)
	bash scripts/bench.sh

profile: $(BIN)
	@echo "=== nsys profile (european, 10M paths) ==="
	nsys profile --stats=true --force-overwrite=true -o data/european_nsys \
	    $(BIN) --option configs/option_european_call.txt --sim configs/sim_default.txt --quiet
	@echo "=== ncu analysis (mc_kernel, asian, 1M paths) ==="
	ncu --set full --launch-count 1 --kernel-name regex:mc_kernel -o data/asian_ncu \
	    $(BIN) --option configs/option_asian_call.txt --sim configs/sim_bench.txt --quiet

clean:
	rm -rf $(BUILD)

# 代码格式化（默认配置 clang-format，需安装：sudo apt install clang-format）
format:
	clang-format -i src/*.h src/*.cpp src/*.cu tests/test_units.cpp
