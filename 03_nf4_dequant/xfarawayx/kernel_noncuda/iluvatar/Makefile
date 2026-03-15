ILCC ?= clang++
TARGET ?= nf4_dequant_iluvatar
SRC ?= main.cu

CXXFLAGS ?= -O3 -std=c++17 -fPIC
INCLUDES ?= -I. -I/usr/local/corex/include
LDFLAGS ?= -L/usr/local/corex/lib64
LDLIBS ?= -lcudart

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) nf4_dequant_kernel.cuh
	$(ILCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]"

clean:
	rm -f $(TARGET)
