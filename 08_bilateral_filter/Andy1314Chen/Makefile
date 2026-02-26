CXX = g++
NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -gencode arch=compute_89,code=sm_89 -I./include

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Detect OpenCV CUDA module (cudaimgproc)
HAVE_OPENCV_CUDA := $(shell pkg-config --libs opencv4 2>/dev/null | grep -q cudaimgproc && echo 1)
ifeq ($(HAVE_OPENCV_CUDA),1)
  CXXFLAGS += -DHAVE_OPENCV_CUDA
  $(info OpenCV CUDA modules detected: enabling cv::cuda::bilateralFilter baseline)
else
  $(info OpenCV CUDA modules not found: cv::cuda baseline disabled)
endif

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = .

SRCS_CPP = $(SRC_DIR)/main.cpp \
           $(SRC_DIR)/image_io.cpp \
           $(SRC_DIR)/bilateral_filter_cpu.cpp \
           $(SRC_DIR)/bilateral_filter_opencv.cpp

SRCS_CUDA = $(SRC_DIR)/bilateral_filter_cuda.cu

OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS_CUDA = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS_CUDA))

TARGET = $(BIN_DIR)/bilateral_filter

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(TARGET): $(OBJS_CPP) $(OBJS_CUDA)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -o $@ $^ $(OPENCV_LIBS) $(CUDA_LIBS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
