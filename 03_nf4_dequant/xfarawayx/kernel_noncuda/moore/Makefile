MCC ?= mcc
TARGET ?= nf4_dequant_musa
SRC ?= main.mu

CXXFLAGS ?= -O3 -std=c++17
INCLUDES ?= -I.
LDFLAGS ?=
LDLIBS ?= -lmusart

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) nf4_dequant_kernel.mu
	$(MCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <weight_file> <output_file> [bf16|fp16] [warmup] [repeats]"

clean:
	rm -f $(TARGET)
