#!/usr/bin/env python3
"""Generate synthetic vector database / query files used by vsearch.

Binary layout is documented in README and implemented in src/binary_io.cpp:

    i32 magic ("VSCH"), i32 version, i64 n, i32 dim,
    u8 len + dtype, u8 len + metric, raw row-major payload (fp32/fp16).

Example:
    python gen_dataset.py --out data \
        --n 1000000 --dim 128 --nq 1000 --top-k 100
"""

import argparse
import os
import struct

import numpy as np


MAGIC = 0x56534348
VERSION = 1


def write_vector_file(path, vectors, metric, dtype="fp32"):
    n, dim = vectors.shape
    with open(path, "wb") as f:
        # Header mirrors include/vsearch/binary_io.hpp:
        #   i32 magic, i32 version, i64 n, i32 dim,
        #   u8 len + dtype, u8 len + metric, raw row-major payload.
        f.write(struct.pack("<iiqi", MAGIC, VERSION, n, dim))
        dt = dtype.encode("ascii")
        mt = metric.encode("ascii")
        f.write(bytes([len(dt)]) + dt)
        f.write(bytes([len(mt)]) + mt)
        if dtype == "fp16":
            vectors.astype(np.float16).tofile(f)
        else:
            vectors.astype(np.float32).tofile(f)
    print(f"wrote {path}: n={n} dim={dim} dtype={dtype} metric={metric}")


def generate(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.clusters:
        cluster_count = args.clusters
    else:
        cluster_count = int(min(max(16, args.n // 2000), 2000))
    centers = rng.normal(size=(cluster_count, args.dim))
    if args.metric in ("cosine",):
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Each vector is a perturbed cluster centroid. Queries are perturbed copies
    # of held-out database vectors, so true neighbors are well defined.
    assign = rng.integers(0, cluster_count, size=args.n, dtype=np.int64)
    vectors = centers[assign] + rng.normal(
        scale=args.vector_scale, size=(args.n, args.dim)
    )

    held = rng.choice(args.n, size=args.nq, replace=args.nq > args.n)
    queries = vectors[held] + rng.normal(
        scale=args.query_scale, size=(args.nq, args.dim)
    )
    query_ids = held.astype(np.int64)

    if args.metric == "cosine":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    write_vector_file(
        os.path.join(args.out, args.db_file), vectors, args.metric, args.dtype
    )
    write_vector_file(
        os.path.join(args.out, args.query_file), queries, "none", args.dtype
    )

    params = f"""# vsearch retrieval parameters
top_k = {args.top_k}
search_mode = "{args.search_mode}"
batch_size = {args.batch_size}
nlist = {args.nlist}
nprobe = {args.nprobe}
pq_m = {args.pq_m}

# extension knobs
kmeans_sample = {args.kmeans_sample}
kmeans_iters = 12
index_path = {os.path.join(args.out, 'index.idx')}
result_path = {os.path.join(args.out, 'result.txt')}
perf_log_path = {os.path.join(args.out, 'perf.log')}
quality_log_path = {os.path.join(args.out, 'quality.log')}
"""
    param_path = os.path.join(args.out, args.params_file)
    with open(param_path, "w", encoding="utf-8") as f:
        f.write(params)
    np.save(os.path.join(args.out, "query_ground_truth_ids.npy"), query_ids)
    print(f"wrote {param_path}")
    print(f"queries are perturbed copies of rows {query_ids[:10]}...")


def main():
    p = argparse.ArgumentParser(description="synthetic vsearch datasets")
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--pq-m", type=int, default=16)
    p.add_argument("--metric", default="l2", choices=["l2", "inner_product", "cosine"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--search-mode", default="ivf_flat",
                   choices=["exact", "ivf_flat", "ivf_pq"])
    p.add_argument("--vector-scale", type=float, default=0.25)
    p.add_argument("--query-scale", type=float, default=0.05)
    p.add_argument("--clusters", type=int, default=0,
                   help="explicit cluster count (0 = auto)")
    p.add_argument("--kmeans-sample", type=int, default=131072)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--db-file", default="vectors.bin")
    p.add_argument("--query-file", default="queries.bin")
    p.add_argument("--params-file", default="params.txt")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
