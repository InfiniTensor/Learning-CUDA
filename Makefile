CUDA_DEVICE ?= 0
PYTHON ?= python
DTYPE ?= bf16
PLATFORM ?= nvidia

# Usage:
#   make nf4 PLATFORM=nvidia
#   make nf4 PLATFORM=nvidia DTYPE=fp16
#   make nf4 PLATFORM=metax
#   make nf4 PLATFORM=moore
#   make nf4 PLATFORM=metax bf16
#   make nf4 PLATFORM=moore fp16

NF4_DIR := 03_nf4_dequant/ikko
NF4_VERIFY := $(NF4_DIR)/verify_mae.py

ifneq ($(filter fp16,$(MAKECMDGOALS)),)
DTYPE := fp16
endif
ifneq ($(filter bf16,$(MAKECMDGOALS)),)
DTYPE := bf16
endif

ifneq ($(filter $(DTYPE),bf16 fp16),$(DTYPE))
$(error Invalid DTYPE='$(DTYPE)'. Use bf16 or fp16)
endif

# =========================================================
# Platform selection
# =========================================================
ifeq ($(PLATFORM),nvidia)
    CC := /usr/local/cuda/bin/nvcc
    SRC_SUFFIX := cu
    CFLAGS := -O3 -std=c++17 -arch=sm_80
    EXTRA_LIBS :=
    NF4_BIN := $(NF4_DIR)/mainla
    RUN_ENV := CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE)

else ifeq ($(PLATFORM),metax)
    CC := mxcc
    SRC_SUFFIX := maca
    CFLAGS := -O3 -std=c++17
    EXTRA_LIBS :=
    NF4_BIN := $(NF4_DIR)/mainla_maca
    RUN_ENV :=

else ifeq ($(PLATFORM),moore)
    CC := mcc
    SRC_SUFFIX := mu
    CFLAGS := -O3 -std=c++11
    EXTRA_LIBS := -I/usr/local/musa/include \
                  -L/usr/lib/gcc/x86_64-linux-gnu/11/ \
                  -L/usr/local/musa/lib \
                  -lmusart
    NF4_BIN := $(NF4_DIR)/mainla_mu
    RUN_ENV :=

else ifeq ($(PLATFORM),iluvatar)
    CC := clang++
    SRC_SUFFIX := cu
    CFLAGS := -O3 -std=c++17
    EXTRA_LIBS := -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
    NF4_BIN := $(NF4_DIR)/mainla_iluvatar
    RUN_ENV := CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE)

else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, metax, moore, iluvatar))
endif

NF4_SRC := $(NF4_DIR)/mainla.$(SRC_SUFFIX)

ifeq ($(DTYPE),fp16)
NF4_OUTPUT := $(NF4_DIR)/data/output_fp16.bin
else
NF4_OUTPUT := $(NF4_DIR)/data/output.bin
endif

.PHONY: nf4 run verify clean help bf16 fp16

help:
	@echo "Usage:"
	@echo "  make nf4 PLATFORM=nvidia"
	@echo "  make nf4 PLATFORM=nvidia DTYPE=fp16"
	@echo "  make nf4 PLATFORM=metax"
	@echo "  make nf4 PLATFORM=moore"
	@echo "  make run PLATFORM=metax"
	@echo "  make verify"

nf4: $(NF4_BIN)
	@echo "=== [NF4] Running $(NF4_BIN) ==="
	@echo "=== PLATFORM=$(PLATFORM), DTYPE=$(DTYPE) ==="
	$(RUN_ENV) ./$(NF4_BIN) $(DTYPE) $(NF4_OUTPUT)
	@echo "=== [NF4] Verifying MAE ==="
	$(PYTHON) $(NF4_VERIFY)

$(NF4_BIN): $(NF4_SRC)
	@echo "=== [NF4] Compiling $(NF4_SRC) with $(CC) ==="
	@mkdir -p .tmp
	TMPDIR=$(CURDIR)/.tmp $(CC) $(CFLAGS) $(NF4_SRC) -o $(NF4_BIN) $(EXTRA_LIBS)

run:
	@echo "=== [NF4] Running $(NF4_BIN) ==="
	@echo "=== PLATFORM=$(PLATFORM), DTYPE=$(DTYPE) ==="
	$(RUN_ENV) ./$(NF4_BIN) $(DTYPE) $(NF4_OUTPUT)

verify:
	@echo "=== [NF4] Verifying MAE ==="
	$(PYTHON) $(NF4_VERIFY)

clean:
	rm -f $(NF4_DIR)/mainla $(NF4_DIR)/mainla_maca $(NF4_DIR)/mainla_mu $(NF4_DIR)/mainla_iluvatar

bf16 fp16:
	@: