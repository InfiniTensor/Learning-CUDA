MXCC ?= mxcc
TARGET ?= nf4_dequant_maca
SRC ?= main.maca

CXXFLAGS ?= -O3 -std=c++17
INCLUDES ?= -I.

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) nf4_dequant_kernel.maca
	$(MXCC) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]"

clean:
	rm -f $(TARGET)
