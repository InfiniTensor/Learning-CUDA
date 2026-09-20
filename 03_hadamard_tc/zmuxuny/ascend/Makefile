BUILD_DIR ?= build/ascend
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
CXX ?= g++
BISHENG := $(ASCEND_HOME)/bin/bisheng
ASCINC := $(ASCEND_HOME)/compiler/tikcpp/tikcfw
ASCFLAGS := -O2 -gline-tables-only -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec -ffp-contract=off -I$(ASCINC) -I$(ASCINC)/interface -I$(ASCINC)/impl -I$(ASCEND_HOME)/include
ASCFLAGS += $(ASCEND_EXTRA_FLAGS)
ASCLIBS := -L$(ASCEND_HOME)/lib64 -lascendcl -lruntime
TARGET ?= hadamard
all: $(BUILD_DIR)/$(TARGET)
$(BUILD_DIR)/libcube.so: ascend/cube.cpp
	mkdir -p $(BUILD_DIR)
	$(BISHENG) $(subst dav-c220-vec,dav-c220-cube,$(ASCFLAGS)) -fPIC -shared $< -o $@ $(ASCLIBS)
$(BUILD_DIR)/libcamp.so: ascend/kernels.cpp ascend/platform.hpp include/numeric.cuh
	mkdir -p $(BUILD_DIR)
	$(BISHENG) $(ASCFLAGS) -fPIC -shared $< -o $@ $(ASCLIBS)
$(BUILD_DIR)/$(TARGET): src/main.cu $(BUILD_DIR)/libcamp.so $(BUILD_DIR)/libcube.so $(wildcard ascend/*.hpp) $(wildcard include/*)
	$(CXX) -x c++ -std=c++17 -O2 -ffp-contract=off -DLP_ASCEND -I$(ASCEND_HOME)/include $< -o $@ -L$(BUILD_DIR) -lcamp -lcube $(ASCLIBS) -Wl,-rpath,'$$ORIGIN'

.PHONY: test numeric-test clean
$(BUILD_DIR)/validate_numeric: tests/validate_numeric.cu $(BUILD_DIR)/libcamp.so $(wildcard ascend/*.hpp) $(wildcard include/*)
	$(CXX) -x c++ -std=c++17 -O2 -ffp-contract=off -DLP_ASCEND -I$(ASCEND_HOME)/include $< -o $@ -L$(BUILD_DIR) -lcamp $(ASCLIBS) -Wl,-rpath,'$$ORIGIN'
numeric-test: $(BUILD_DIR)/validate_numeric
	$(BUILD_DIR)/validate_numeric
test: all numeric-test
	python3 tests/validate.py --binary $(BUILD_DIR)/$(TARGET) --output results/ascend/correctness.json
clean:
	rm -rf $(BUILD_DIR)
