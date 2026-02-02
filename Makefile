# *********************************************************************
# Learning-CUDA Makefile
# Targets:
#   make               		: Build + run tests (default, non-verbose)
#   make build         		: Only compile (no run)
#   make run           		: Run tests (after build, non-verbose)
#   make run VERBOSE=true 	: Run tests with verbose output
#   make clean         		: Delete temporary files
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM        ?= nvidia
PLATFORM_DEFINE ?= -DPLATFORM_NVIDIA
STUDENT_SUFFIX  := cu
CFLAGS          := -std=c++17 -O0
EXTRA_LIBS     	:= 

# Compiler & Tester object selection based on PLATFORM
ifeq ($(PLATFORM),nvidia)
    CC          	:= nvcc
    TEST_OBJ    	:= tester/tester_nv.o
	PLATFORM_DEFINE := -DPLATFORM_NVIDIA
else ifeq ($(PLATFORM),iluvatar)
    CC          	:= clang++
	CFLAGS          := -std=c++17 -O3
    TEST_OBJ    	:= tester/tester_iluvatar.o
	PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
	EXTRA_LIBS		:= -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
else ifeq ($(PLATFORM),moore)
    CC          	:= mcc
	CFLAGS          := -std=c++11 -O3
    TEST_OBJ    	:= tester/tester_moore.o
	STUDENT_SUFFIX  := mu
	PLATFORM_DEFINE := -DPLATFORM_MOORE
	EXTRA_LIBS		:= -I/usr/local/musa/include -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
else ifeq ($(PLATFORM),metax)
    CC          	:= mxcc
    TEST_OBJ    	:= tester/tester_metax.o
	STUDENT_SUFFIX  := maca
	PLATFORM_DEFINE := -DPLATFORM_METAX
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif

# Executable name
TARGET          	:= test_kernels
# Kernel implementation
STUDENT_SRC     	:= src/kernels.$(STUDENT_SUFFIX) 
# Compiled student object (auto-generated)
STUDENT_OBJ  		:= $(addsuffix .o,$(basename $(STUDENT_SRC)))
# Tester's actual verbose argument (e.g., --verbose, -v)
TEST_VERBOSE_FLAG 	:= --verbose
# User-provided verbose mode (true/false; default: false)
VERBOSE         	:=  

# -------------------------------
# Process User Input (VERBOSE → Tester Flag)
# -------------------------------
# Translates `VERBOSE=true` (case-insensitive) to the tester's verbose flag.
# If VERBOSE is not "true" (or empty), no flag is passed.
VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG), )

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: all build run clean

# Default target: Build + run tests (non-verbose)
all: build run

# Build target: Compile student code + link with test logic
build: $(TARGET)

# Run target: Execute tests (supports `VERBOSE=true` for verbose output)
run: $(TARGET)
	@echo "=== Running tests (output from $(STUDENT_OBJ)) ==="
	@# Show verbose mode status (friendly for users)
	@if [ -n "$(VERBOSE_ARG)" ]; then \
	    echo "=== Verbose mode: Enabled (using '$(TEST_VERBOSE_FLAG)') ==="; \
	else \
	    echo "=== Verbose mode: Disabled ==="; \
	fi
	./$(TARGET) $(VERBOSE_ARG)

# Clean target: Delete temporary files (executable + src object)
clean:
	@echo "=== Cleaning temporary files ==="
	rm -f $(TARGET) $(STUDENT_OBJ)

# -------------------------------
# Dependency Rules (Core Logic)
# -------------------------------
# Generate executable: Link kernel code (kernels.o) with test logic (tester.o)
$(TARGET): $(STUDENT_OBJ) $(TEST_OBJ)
	@echo "=== Linking executable (student code + test logic) ==="
ifeq ($(PLATFORM),nvidia)
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ --compiler-options -lstdc++
else
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ $(EXTRA_LIBS)
endif

# Generate src object: Compile kernels.cu (triggers template instantiation)
$(STUDENT_OBJ): $(STUDENT_SRC)
	@echo "=== Compiling student code ($(STUDENT_SRC)) ==="
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -c $< -o $@
