NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS  = -arch=$(ARCH) -O3 -std=c++17 --expt-relaxed-constexpr -I. -Ifht
LDFLAGS    = -lcurand

SRCS = bench.cu hadamard.cu hadamard_hybrid.cu hadacore.cu fht/fast_hadamard_transform_cuda.cu
OBJS = $(SRCS:.cu=.o)

bench: $(OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $(OBJS) $(LDFLAGS)

fuse_bench: fuse/fuse_bench.o fuse/fuse_per_row_quant.o fuse/fuse_per_group_quant.o hadamard.o
	$(NVCC) -arch=$(ARCH) -o $@ $^ $(LDFLAGS)

fuse_group_bench: fuse/fuse_group_bench.o fuse/fuse_per_group_quant.o hadamard.o
	$(NVCC) -arch=$(ARCH) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) fuse/fuse_bench.o fuse/fuse_per_row_quant.o \
		fuse/fuse_per_group_quant.o fuse/fuse_group_bench.o \
		bench fuse_bench fuse_group_bench

.PHONY: clean
