# Learning-CUDA nf4_dequant Makefile
# Target platforms supported: nvidia (default), iluvatar, moore, metax

PLATFORM        ?= nvidia
PLATFORM_DEFINE ?= -DPLATFORM_NVIDIA
STUDENT_SUFFIX  := cu
CFLAGS          := -std=c++17 -O3
EXTRA_LIBS      :=

ifeq ($(PLATFORM),nvidia)
    CC                  := nvcc
    PLATFORM_DEFINE     := -DPLATFORM_NVIDIA
    CFLAGS              += -lineinfo --ptxas-options=-v -use_fast_math -arch=sm_80
else ifeq ($(PLATFORM),iluvatar)
    CC                  := clang++
    PLATFORM_DEFINE     := -DPLATFORM_ILUVATAR
    EXTRA_LIBS          := -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
else ifeq ($(PLATFORM),moore)
    CC                  := mcc
    STUDENT_SUFFIX      := mu
    PLATFORM_DEFINE     := -DPLATFORM_MOORE
    EXTRA_LIBS          := -I/usr/local/musa/include -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
else ifeq ($(PLATFORM),metax)
    CC                  := mxcc
    STUDENT_SUFFIX      := maca
    PLATFORM_DEFINE     := -DPLATFORM_METAX
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif

TARGET := nf4_dequantizer
MAIN_SRC := main.$(STUDENT_SUFFIX)
KERNEL_SRC := src/dequantize.$(STUDENT_SUFFIX)

.PHONY: all build run clean

all: build run

build: $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

$(TARGET): $(MAIN_SRC) $(KERNEL_SRC)
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ $(EXTRA_LIBS)
