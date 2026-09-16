# 2026-09-08 validation commands

The remote host address and credentials are intentionally not recorded.

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 8
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/unit_tests
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/ivf_flat_tests
```

```bash
python3 scripts/generate_data.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --num-vectors 1000000 --num-queries 1000 --dim 128 \
  --dtype fp32 --metric inner_product --seed 2026

./build/ivf_build \
  --database data/million/inner_fp32_db.bin \
  --output data/million/inner_fp32_nlist256.ivf \
  --nlist 256 --iterations 15 --training-samples 100000 --threads 16

python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k100_batch64.conf \
  --exact results/exact_k100.txt \
  --output-dir results/k100_formal --csv results/k100_formal.csv \
  --nlist 256 --nprobes 128,160,192,224 \
  --training-samples 100000 --training-iterations 15 \
  --warmup 5 --repeat 100
```

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_warp_compact_batch40.conf \
  --output results/k50_batch40.txt \
  --backend gpu --warmup 1 --repeat 100

./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_batch64.conf \
  --output results/k50_batch64.txt \
  --backend gpu --warmup 1 --repeat 100

python3 scripts/evaluate_recall.py \
  --exact results/new_gpu/exact_k50.txt \
  --approximate results/k50_batch40.txt
```

```bash
./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_warp_compact_batch40.conf \
  --output results/k50_probe_major.txt \
  --backend gpu --warmup 5 --repeat 100

./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k50_bucket_major.conf \
  --output results/k50_bucket_major.txt \
  --backend gpu --warmup 5 --repeat 100

cmp results/k50_probe_major.txt results/k50_bucket_major.txt
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/ivf_flat_tests
```

K=100 heap and combined bucket-major A/B used the retained pre-optimization
binary and the rebuilt binary on the same host:

```bash
./build/ivf_search_pre_k100_heap \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k100_warp1.conf \
  --output results/k100_pre_heap.txt \
  --backend gpu --warmup 5 --repeat 100

./build/ivf_search \
  --index data/million/inner_fp32_nlist256.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_k100_bucket_major.conf \
  --output results/k100_heap_bucket_major.txt \
  --backend gpu --warmup 5 --repeat 100

cmp results/k100_pre_heap.txt results/k100_heap_bucket_major.txt
```

Rejected K=100 selection prototypes were tested against the retained stable
binary on the same host. The buffer-8 prototype was stopped after 10 repeats
because its QPS regression exceeded 20%. Kernel resources were inspected from
the compiled object before restoring the retained source.

```bash
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  build/CMakeFiles/vector_engine.dir/src/gpu_ivf_flat.cu.o

sha256sum src/gpu_ivf_flat.cu
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

The same-batch FAISS comparison reused the host CUDA 12.8 libraries. The venv
contained FAISS 1.14.1 and NumPy 2.2.6; disabling wheel preloading avoids
installing a second copy of CUDA runtime and cuBLAS.

```bash
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

/data/purplepulse-faiss-system-venv/bin/python scripts/benchmark_faiss.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --output-dir results/faiss_batch40 \
  --csv results/faiss_batch40.csv \
  --index-dir /data/purplepulse-faiss-indexes \
  --reference-results results/new_gpu/exact_k50.txt \
  --backends gpu --index-types ivf_flat \
  --top-ks 50 --nprobes 160 --nlist 256 --batch-size 40 \
  --train-samples 100000 --iterations 15 --seed 2026 \
  --warmup 5 --repeat 100 --cpu-threads 1
```

```bash
python3 scripts/import_mnist.py \
  --input data/mnist/mnist.npz \
  --database data/mnist/database.bin \
  --queries data/mnist/queries.bin

./build/ivf_build \
  --database data/mnist/database.bin --output data/mnist/nlist256.ivf \
  --nlist 256 --iterations 15 --training-samples 60000 --threads 16

./build/vector_search \
  --database data/mnist/database.bin --queries data/mnist/queries.bin \
  --params configs/exact_million_k10_fused_batch64.conf --backend gpu \
  --output results/mnist_exact_k10.txt --warmup 2 --repeat 10

python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/mnist/nlist256.ivf --queries data/mnist/queries.bin \
  --params configs/ivf_flat_mnist.conf \
  --exact results/mnist_exact_k10.txt \
  --output-dir results/mnist_ivf --csv results/mnist_ivf.csv \
  --nlist 256 --nprobes 8,16,32,64,96,128 \
  --training-samples 60000 --training-iterations 15 \
  --warmup 2 --repeat 10
```

The final source-only audit excluded generated artifacts before configuring a new
directory:

```bash
tar --exclude='./build' --exclude='./data' --exclude='./results' \
  --exclude='./evidence' --exclude='./.git' \
  -cf /data/purplepulse-submit-audit.tar .
mkdir /data/purplepulse-submit-audit
tar -xf /data/purplepulse-submit-audit.tar \
  -C /data/purplepulse-submit-audit
cmake -S /data/purplepulse-submit-audit \
  -B /data/purplepulse-submit-audit/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /data/purplepulse-submit-audit/build -j 16
ctest --test-dir /data/purplepulse-submit-audit/build --output-on-failure
```
