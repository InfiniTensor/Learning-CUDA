WDIR = .
SRC_DIR = $(WDIR)/src
OBJ_DIR = $(WDIR)/obj
INCLUDES_DIR = $(WDIR)/include
ANALYSIS_DIR = $(WDIR)/analysis
ENV_DIR = $(WDIR)/tmp_env
SCRIPTS_DIR = $(WDIR)/scripts
MK = $(WDIR)/init.mk
RESULT_DIR = ./result
LOG = $(RESULT_DIR)/runtime.log

-include $(MK)

PLATFORM        ?= nvidia
# resolve platform
ifeq ($(PLATFORM),nvidia)
    PLATFORM_DEFINE 	:= -DPLATFORM_NVIDIA
    PREFIX 		:= /usr/local/cuda-12.8/bin/
    CXX 		:= $(PREFIX)nvcc
    GDB 		:= $(PREFIX)cuda-gdb
    NCU 		:= $(PREFIX)ncu
    NCU-GUI 		:= $(PREFIX)ncu-ui
    NSYS 		:= $(PREFIX)nsys
    CXXFLAGS		:= -Xcompiler -std=c++17 -O3 -use_fast_math -Xcudafe --diag_suppress=611 -Wno-deprecated-gpu-targets -I$(INCLUDES_DIR)
    SUFFIX		:= cu
else ifeq ($(PLATFORM),iluvatar)
    PLATFORM_DEFINE 	:= -DPLATFORM_ILUVATAR
    CXX          	:= clang++
    CXXFLAGS	        := -std=c++17 -O3 -I$(INCLUDES_DIR) -Wno-implicit-const-int-float-conversion -Wno-literal-range
    LIBS		:= -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
    SUFFIX  		:= cu
else ifeq ($(PLATFORM),moore)
    PLATFORM_DEFINE 	:= -DPLATFORM_MOORE
    CXX          	:= mcc
    CXXFLAGS          	:= -std=c++17 -O3 -I$(INCLUDES_DIR) -I/usr/local/musa/include 
    LIBS		:= -L$(ENV_DIR)/lib -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
    RUNTIME_LIBS	+= /usr/local/musa/lib
    SUFFIX  		:= mu
else ifeq ($(PLATFORM),metax)
    PLATFORM_DEFINE 	:= -DPLATFORM_METAX
    CXX          	:= mxcc
    CXXFLAGS		:= -std=c++17 -O3 -I$(INCLUDES_DIR)
    SUFFIX  		:= maca
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif
-include $(SCRIPTS_DIR)/$(PLATFORM).mk


ifeq ($(HAS_CV_ENV), yes)
    CXXFLAGS += -DHAS_CV	
endif



SRCS = $(wildcard $(SRC_DIR)/*.$(SUFFIX))
OBJS = $(patsubst $(SRC_DIR)/%.$(SUFFIX), $(OBJ_DIR)/%.o, $(SRCS))
EXEC = $(WDIR)/runTester


.DEFAULT_GOAL := all

all: $(EXEC)


init:
	@rm -rf $(ENV_DIR) $(MK)
	@echo "[INIT] Checking OpenCV Environment..."
	@$(PLATFORM_INIT_ENV); \
	if pkg-config --exists opencv4; then \
		echo "[INFO] Built-in OpenCV Found."; \
		echo "HAS_CV_ENV = yes" > $(MK); \
		echo "CVFLAGS = $$(pkg-config --cflags opencv4)" >> $(MK); \
		echo "CVLIBS = $$(pkg-config --libs opencv4)" >> $(MK); \
		echo "CVRUNTIME_LIBS = $${LD_LIBRARY_PATH}" >> $(MK); \
	elif command -v conda > /dev/null 2>&1; then \
		echo "[INFO] Using Conda to Create $(ENV_DIR)..."; \
		conda create --prefix $(ENV_DIR) -y libopencv; \
		echo "HAS_CV_ENV = yes" > $(MK); \
		echo "CVFLAGS = -I$(ENV_DIR)/include/opencv4" >> $(MK); \
		echo "CVLIBS = -L$(ENV_DIR)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" >> $(MK); \
		echo "CVRUNTIME_LIBS = $(ENV_DIR)/lib:$${LD_LIBRARY_PATH}" >> $(MK); \
	else \
		echo "[WARN] No OpenCV is Available on this Server."; \
		echo "HAS_CV_ENV = no" > $(MK); \
	fi
	@echo "[INFO] Project Init Done."
	@echo "[INFO] Enter \`make (run) PLATFORM=[nvidia, moore, metax, iluvatar]\` to Compile & Run."




#link: .o -> exec
$(EXEC): $(OBJS)
	@echo "[INFO] $^ -> $@"
	@$(CXX) $(CXXFLAGS) $(CVFLAGS) $(LIBS) $(CVLIBS) -o $@ $^


#compile: .cu -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.$(SUFFIX)
	@mkdir -p $(OBJ_DIR)
	@echo "[INFO] $< -> $@"
	@$(CXX) $(CXXFLAGS) $(CVFLAGS) $(PLATFORM_DEFINE) -c -o $@ $<


run: $(EXEC)
	@mkdir -p $(RESULT_DIR)
	@export LD_LIBRARY_PATH=$(CVRUNTIME_LIBS):$$LD_LIBRARY_PATH; \
	$(MAKE) -s -f $(SCRIPTS_DIR)/$(PLATFORM).mk exec LD_LIBRARY_PATH="$$LD_LIBRARY_PATH" 2>&1 | tee $(LOG)
	@echo "[INFO] Runtime log has been written to $(LOG)"
	


gdb: $(EXEC)
	@$(GDB) $<




#Nsight System,系统级分析工具
nsys: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NSYS) profile -t cuda,nvtx,osrt -o $(ANALYSIS_DIR)/$(EXEC) -f true $<
	@$(NSYS) stats $(ANALYSIS_DIR)/$(EXEC).nsys-rep --force-export=true

#Nsight Compute,内核级分析工具
ncu: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NCU) --print-details all --nvtx --call-stack --set full $<

ncu-gui: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NCU) --nvtx --call-stack --set full -f --export $(ANALYSIS_DIR)/$(EXEC).ncu-rep $<
	@$(NCU-GUI) $(ANALYSIS_DIR)/$(EXEC).ncu-rep


clean:
	rm -rf $(OBJ_DIR) $(ANALYSIS_DIR) $(EXEC) $(RESULT_DIR)

cleanup:
	rm -rf $(OBJ_DIR) $(ANALYSIS_DIR) $(EXEC) $(RESULT_DIR)
	rm -rf $(MK) $(ENV_DIR)


.PHONY:
