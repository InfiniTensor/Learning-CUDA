# 2026-09-04 正式验收命令

所有命令均在项目根目录执行。登录信息、主机地址和密码不属于实验口径，未写入
证据文件。

## 构建与测试

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/unit_tests
```

## 数据与自研索引

```bash
python3 scripts/generate_data.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --num-vectors 1000000 --num-queries 1000 --dim 128 \
  --dtype fp32 --metric inner_product --seed 2026

sha256sum data/million/inner_fp32_db.bin data/million/inner_fp32_q.bin

./build/ivf_build \
  --database data/million/inner_fp32_db.bin \
  --output data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --nlist 256 --iterations 15 --training-samples 100000

sha256sum data/million/inner_fp32_nlist256_spherical_100k15.ivf
```

## PurplePulse Exact

```bash
for k in 10 50 100; do
  ./build/vector_search \
    --database data/million/inner_fp32_db.bin \
    --queries data/million/inner_fp32_q.bin \
    --params configs/exact_million_k${k}_fused_batch64.conf \
    --backend gpu \
    --output results/formal_100k15/exact_k${k}.txt \
    --warmup 5 --repeat 100
done
```

## PurplePulse IVF

K=10/50/100 分别使用
`ivf_flat_million_batch64.conf`、`ivf_flat_million_k50_batch64.conf` 和
`ivf_flat_million_k100_batch64.conf`。以下是 K=10 命令，其余两项只替换
配置、Exact 文件、输出目录和 CSV 名称。

```bash
python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_batch64.conf \
  --exact results/formal_100k15/exact_k10.txt \
  --output-dir results/formal_100k15/ivf_k10 \
  --csv results/formal_100k15/ivf_k10.csv \
  --nlist 256 --nprobes 128,160,192,224 \
  --training-samples 100000 --training-iterations 15 \
  --warmup 5 --repeat 100
```

## FAISS GPU

```bash
OMP_NUM_THREADS=1 python3 scripts/benchmark_faiss.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --reference-results results/formal_100k15/exact_k100.txt \
  --index-dir data/million/faiss_formal_100k15 \
  --output-dir results/formal_100k15/faiss_gpu \
  --csv results/formal_100k15/faiss_gpu.csv \
  --backends gpu --index-types flat,ivf_flat \
  --top-ks 10,50,100 --nlist 256 --nprobes 128,160,192,224 \
  --batch-size 64 --train-samples 100000 --iterations 15 --seed 2026 \
  --warmup 5 --repeat 100 --cpu-threads 1 --rebuild
```

## FAISS CPU Exact

```bash
OMP_NUM_THREADS=1 python3 scripts/benchmark_faiss.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --reference-results results/formal_100k15/exact_k100.txt \
  --index-dir data/million/faiss_formal_100k15 \
  --output-dir results/formal_100k15/faiss_cpu \
  --csv results/formal_100k15/faiss_cpu_flat_repeat20.csv \
  --backends cpu --index-types flat --top-ks 10,50,100 \
  --nlist 256 --nprobes 160 --batch-size 64 \
  --train-samples 100000 --iterations 15 --seed 2026 \
  --warmup 1 --repeat 20 --cpu-threads 1
```

## Nsight Systems 代表性采样

```bash
nsys profile --force-overwrite true --stats=true \
  -o results/evidence/exact_k10_nsys \
  ./build/vector_search \
    --database data/million/inner_fp32_db.bin \
    --queries data/million/inner_fp32_q.bin \
    --params configs/exact_million_k10_fused_batch64.conf \
    --backend gpu \
    --output results/formal_100k15/exact_k10_nsys.txt \
    --warmup 1 --repeat 5
```
