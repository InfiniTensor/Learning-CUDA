# Particle simulation in uniform magnetic field (CUDA)
# Requires: CUDA Toolkit 11+, C++17

NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2
TARGET = particle_sim

SRCS = main.cu
OBJS = main.o

.PHONY: all clean run format

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) $(OBJS) *.bin

run: $(TARGET)
	./$(TARGET) data/particles.txt data/field.txt data/params.txt out.bin

format:
	clang-format -i main.cu
