# *********************************************************************
# Learning-CUDA Makefile
# Targets:
#   make               : Build + run tests (default, non-verbose)
#   make build         : Only compile (no run)
#   make run           : Run tests (after build, non-verbose)
#   make run VERBOSE=true : Run tests with verbose output
#   make clean         : Delete temporary files
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
CC              := nvcc                  # CUDA compiler
CFLAGS          := -std=c++17 -O0        # Compile flags
TARGET          := test_kernels     	 # Executable name
STUDENT_SRC     := src/kernels.cu        # Kernel implementation 
STUDENT_OBJ     := $(STUDENT_SRC:.cu=.o) # Compiled student object (auto-generated)
TEST_OBJ        := tester/tester.o       # Pre-compiled test object
TEST_VERBOSE_FLAG := --verbose            # Tester's actual verbose argument (e.g., --verbose, -v)
VERBOSE         :=                      # User-provided verbose mode (true/false; default: false)

# -------------------------------
# Process User Input (VERBOSE → Tester Flag)
# -------------------------------
# Translates `VERBOSE=true` (case-insensitive) to the tester's verbose flag.
# If VERBOSE is not "true" (or empty), no flag is passed.
VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG), )

# -------------------------------
# Phony Targets (No Files Generated)
# -------------------------------
.PHONY: all build run clean

# Default target: Build + run tests (non-verbose)
all: build run

# Build target: Compile student code + link with test logic
build: $(TARGET)

# Run target: Execute tests (supports `VERBOSE=true` for verbose output)
run: $(TARGET)
	@echo "=== Running tests (output from tester.o) ==="
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
	$(CC) $(CFLAGS) -o $@ $^

# Generate src object: Compile kernels.cu (triggers template instantiation)
$(STUDENT_OBJ): $(STUDENT_SRC)
	@echo "=== Compiling student code (src/kernels.cu) ==="
	$(CC) $(CFLAGS) -c $< -o $@
