# =============================================================================
#  Makefile -- reference build (Linux).  For the Windows development build use
#  scripts/build.ps1, which drives the same nvcc invocations through vcvars64.
#
#  Targets
#    all      build/fhwt (CLI)
#    check    run the built-in correctness self-check (`--mode verify`: element-wise
#             against a double-precision CPU reference, judged with the project's
#             own tolerances) -- this is the check that ships with the program
#    check-tests  build and run the five test suites (development tree only: the
#             generic project rules ask for "no test code", so tests/ is not part
#             of the submission package)
#    bench    run the full experiment sweep into data/
#    refcheck compare against fast_hadamard_transform (needs PYTHONPATH to it)
#    report   data/*.csv -> docs/结果表格.md
#    clean    remove build artefacts
#
#  Notes
#    * `-arch=native` is used by default; override with
#      `make ARCH=sm_90` (or sm_80 / sm_120 / compute_90 ...) for cross builds.
#      The Tensor Core path needs sm_80+; it is compiled out (empty kernel)
#      below that so the file still assembles.
#    * `--use_fast_math` is deliberately NOT used: it changes IEEE semantics and
#      would invalidate the error measurements against the fp64 reference.
# =============================================================================
ARCH      ?= native
NVCC      ?= nvcc
BUILD     ?= build
SRC_COMMON = src/ops.cu

NVFLAGS   := -std=c++17 -O3 -Iinclude -I. -arch=$(ARCH) \
             -Xcudafe --unicode_source_kind=UTF-8 -lineinfo \
             -diag-suppress 192,177,20012
LDFLAGS   :=

CLI_SRC   = src/main.cu $(SRC_COMMON)
TESTS     = test_correctness test_fusion test_fp8 test_graph test_gemm
TEST_BIN  = $(addprefix $(BUILD)/,$(addsuffix .exe,$(TESTS)))

.PHONY: all check check-tests bench clean info report refcheck

all: $(BUILD)/fhwt.exe

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/fhwt.exe: $(CLI_SRC) include/fhwt/*.cuh include/fhwt/fhwt.h | $(BUILD)
	$(NVCC) $(NVFLAGS) $(CLI_SRC) -o $@ $(LDFLAGS)

$(BUILD)/%.exe: tests/%.cu $(SRC_COMMON) include/fhwt/*.cuh tests/test_util.h | $(BUILD)
	$(NVCC) $(NVFLAGS) $< $(SRC_COMMON) -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------------
#  Correctness / regression
# ---------------------------------------------------------------------------
# The program's own self-check: no test code, just the kernel path against a
# double-precision reference on the CPU, with the tolerances the brief sets
# (1e-2 for fp16, 5e-2 for bf16).  The `--mode quant` lines check the fusion
# requirement: fused output == "rotate, then quantise", bit for bit (bytes and
# scales), per-row and per-block scaling.
check: $(BUILD)/fhwt.exe
	$(BUILD)/fhwt.exe --mode verify --dim 64  --rows 4096
	$(BUILD)/fhwt.exe --mode verify --dim 128 --rows 4096
	$(BUILD)/fhwt.exe --mode verify --dim 256 --rows 4096
	$(BUILD)/fhwt.exe --mode verify --dim 512 --rows 4096
	$(BUILD)/fhwt.exe --mode verify --dim 1024 --rows 4096
	$(BUILD)/fhwt.exe --mode verify --dim 128 --rows 4096 --kernel tc --dtype bf16
	$(BUILD)/fhwt.exe --mode verify --dim 4096 --rows 512 --kernel smem --acc fp32
	$(BUILD)/fhwt.exe --mode quant --quant fp8  --dim 64   --rows 4096 --iters 5 --warmup 1
	$(BUILD)/fhwt.exe --mode quant --quant fp8  --dim 256  --rows 4096 --iters 5 --warmup 1 --dtype bf16
	$(BUILD)/fhwt.exe --mode quant --quant int4 --dim 128  --rows 4096 --iters 5 --warmup 1
	$(BUILD)/fhwt.exe --mode quant --quant int4 --dim 1024 --rows 4096 --iters 5 --warmup 1 --block-size 32
	$(BUILD)/fhwt.exe --mode quant --quant fp8  --dim 512  --rows 4096 --iters 5 --warmup 1 --block-size 128

# Development tree only (tests/ is not part of the submission package).
check-tests: $(TEST_BIN)
	$(BUILD)/test_correctness.exe
	$(BUILD)/test_fusion.exe
	$(BUILD)/test_fp8.exe
	$(BUILD)/test_graph.exe
	$(BUILD)/test_gemm.exe

# ---------------------------------------------------------------------------
#  Experiment sweep (writes CSVs consumed by tools/analyze.py)
# ---------------------------------------------------------------------------
bench: $(BUILD)/fhwt.exe
	@mkdir -p data docs/profile
	$(BUILD)/fhwt.exe --mode info | tee data/device_info.txt
	$(BUILD)/fhwt.exe --mode tune   --rows 65536 --dtype f16 --dims 64,128,256,512,1024 \
	    --iters 50 --warmup 10 --reps 3 --csv data/tune.csv
	$(BUILD)/fhwt.exe --mode matrix --kernel reg  --iters 50 --warmup 10 --reps 3 --csv data/kernels_reg.csv
	$(BUILD)/fhwt.exe --mode matrix --kernel tc   --iters 50 --warmup 10 --reps 3 --csv data/kernels_tc.csv
	$(BUILD)/fhwt.exe --mode matrix --kernel smem \
	    --dims 64,128,256,512,1024,2048,4096,8192,16384 \
	    --batch 2048,8192,32768,131072 --iters 50 --warmup 10 --reps 3 --csv data/kernels_smem.csv
	$(BUILD)/fhwt.exe --mode inplace --dims 128,256,512,1024 --dtypes f16,bf16 \
	    --batch 8192,16384,32768,65536,131072,262144 \
	    --iters 100 --warmup 10 --reps 5 --csv data/inplace.csv
	for q in fp8 int4; do \
	  for d in 64 128 256 512 1024; do \
	    $(BUILD)/fhwt.exe --mode quant --quant $$q --dim $$d --rows 262144 \
	        --dtype f16 --iters 50 --warmup 10 > data/quant_$${q}_$${d}.log; \
	  done; \
	done
	$(BUILD)/fhwt.exe --mode cpu --dim 256 --rows 65536 --threads 8 > data/cpu.log
	# Call-site advice (in-place / batching), report 7.10.
	for sh in 1024x64 4096x128 16384x256 65536x256 131072x256 262144x512; do \
	  $(BUILD)/fhwt.exe --mode advice --rows $${sh%x*} --dim $${sh#*x} --dtype f16; echo; \
	done > data/advice.log
	# CUDA Graph vs raw launches (report 8.5).  The shape list and the
	# equal-duration-burst logic live in tools/bench_graph.py so that this target
	# and scripts/run_experiments.ps1 produce identical numbers.
	python3 tools/bench_graph.py --bin $(BUILD)/fhwt.exe --out data
	# How many launches are inside the graph (report 7.11).  Same reasoning as
	# bench_graph.py: the sweep lives in the tool so both build paths agree.
	python3 tools/bench_loop.py --bin $(BUILD)/fhwt.exe --out data --reps 3 --runs 5
	# How many *roots* the recorded graph has (report 5.8): layers 1..32, each
	# measured all-roots and single-root, so the two columns share a drift.
	python3 tools/bench_roots.py --bin $(BUILD)/fhwt.exe --out data --reps 3 --runs 5
	# Rotation fused into the GEMM's A-tile load (report 7.12).  The shape sweep
	# lives in the tool for the same reason as the two above.
	python3 tools/bench_gemm.py --bin $(BUILD)/fhwt.exe --out data --reps 3 --runs 5
	# Fork/join *width*: N independent tensors per layer, plus the multi-stream
	# control (report 7.13).  --tensors/--group-entry/--streams live in the tool.
	python3 tools/bench_group.py --bin $(BUILD)/fhwt.exe --out data --reps 3 --runs 3
	# How many passes one graph should hold (report 7.14).  K must divide --iters.
	python3 tools/bench_batch.py --bin $(BUILD)/fhwt.exe --out data --reps 3 --runs 8

# ---------------------------------------------------------------------------
#  Report tables
# ---------------------------------------------------------------------------
report: data/kernels_reg.csv data/kernels_tc.csv data/kernels_smem.csv
	python3 tools/analyze.py --data data --out docs/结果表格.md
	python3 tools/profile_summary.py --dir docs/profile

# Element-by-element comparison with the reference implementation named by the
# project brief.  Requires the compiled fast_hadamard_transform on PYTHONPATH.
refcheck: $(BUILD)/fhwt.exe
	python3 tools/check_against_ref.py --bin $(BUILD)/fhwt.exe \
	    --dims 64,128,256,512,1024 --rows 1024 --kernels reg,tc,smem
	python3 tools/check_against_ref.py --bin $(BUILD)/fhwt.exe \
	    --dims 2048,4096,8192,16384 --rows 256 --kernels smem

info: $(BUILD)/fhwt.exe
	$(BUILD)/fhwt.exe --mode info

clean:
	rm -rf $(BUILD)
