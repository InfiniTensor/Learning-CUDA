#!/usr/bin/env python3
"""生成测试用 CSR 格式图（二进制）和查询文件。

用法:
  python3 gen_graph.py <N> <M> <out_graph.csr> <out_queries.txt> [--seed S] [--queries Q] [--type TYPE]

图类型:
  random   - 随机有向图（默认）
  grid     - 2D 网格图（有上下左右边）
  complete - 完全有向图（M = N*(N-1)，可能很大）
"""
import argparse
import random
import struct
import sys


def write_csr(path, N, row_ptr, col_idx, cap):
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, len(col_idx)))
        f.write(struct.pack(f"{len(row_ptr)}i", *row_ptr))
        f.write(struct.pack(f"{len(col_idx)}i", *col_idx))
        f.write(struct.pack(f"{len(cap)}i", *cap))


def gen_random(N, M, seed, cap_max=10000):
    """生成 M 条随机有向边（去重自环，可能有重边）。"""
    rng = random.Random(seed)
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v:
            v = (v + 1) % N
        c = rng.randint(1, cap_max)
        edges[u].append((v, c))

    row_ptr = [0] * (N + 1)
    col_idx = []
    cap = []
    for u in range(N):
        row_ptr[u + 1] = row_ptr[u] + len(edges[u])
        for v, c in edges[u]:
            col_idx.append(v)
            cap.append(c)
    return row_ptr, col_idx, cap


def gen_grid(N_side, seed, cap_max=10000):
    """生成 N_side x N_side 的 2D 网格图，4 邻接。"""
    rng = random.Random(seed)
    N = N_side * N_side
    edges = [[] for _ in range(N)]
    for i in range(N_side):
        for j in range(N_side):
            u = i * N_side + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N_side and 0 <= nj < N_side:
                    v = ni * N_side + nj
                    edges[u].append((v, rng.randint(1, cap_max)))
    row_ptr = [0] * (N + 1)
    col_idx = []
    cap = []
    for u in range(N):
        row_ptr[u + 1] = row_ptr[u] + len(edges[u])
        for v, c in edges[u]:
            col_idx.append(v)
            cap.append(c)
    return N, row_ptr, col_idx, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, help="节点数（grid 模式为每边节点数）")
    ap.add_argument("M", type=int, help="边数（grid 模式忽略）")
    ap.add_argument("graph_out")
    ap.add_argument("queries_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--type", choices=["random", "grid", "complete"], default="random")
    ap.add_argument("--cap-max", type=int, default=10000)
    args = ap.parse_args()

    if args.type == "grid":
        N, row_ptr, col_idx, cap = gen_grid(args.N, args.seed, args.cap_max)
    elif args.type == "complete":
        N = args.N
        edges = [[] for _ in range(N)]
        rng = random.Random(args.seed)
        for u in range(N):
            for v in range(N):
                if u != v:
                    edges[u].append((v, rng.randint(1, args.cap_max)))
        row_ptr = [0] * (N + 1)
        col_idx = []
        cap = []
        for u in range(N):
            row_ptr[u + 1] = row_ptr[u] + len(edges[u])
            for v, c in edges[u]:
                col_idx.append(v)
                cap.append(c)
    else:
        N = args.N
        row_ptr, col_idx, cap = gen_random(N, args.M, args.seed, args.cap_max)

    write_csr(args.graph_out, N, row_ptr, col_idx, cap)
    print(f"[gen] 图: N={N} M={len(col_idx)} -> {args.graph_out}")

    # 生成查询：源点 = 0，汇点 = N-1，以及若干随机对
    rng = random.Random(args.seed + 1)
    queries = []
    queries.append((0, N - 1))
    while len(queries) < args.queries:
        s = rng.randint(0, N - 1)
        t = rng.randint(0, N - 1)
        if s != t and (s, t) not in queries:
            queries.append((s, t))
    with open(args.queries_out, "w") as f:
        for s, t in queries:
            f.write(f"{s} {t}\n")
    print(f"[gen] 查询数={len(queries)} -> {args.queries_out}")


if __name__ == "__main__":
    main()
