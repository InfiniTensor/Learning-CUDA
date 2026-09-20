#!/usr/bin/env python3
"""Reproducible CSR fixtures. No third-party Python packages required."""
import argparse
import array
import json
import random
import struct
import sys
from pathlib import Path


def write_csr(path, n, edges):
    edges.sort(key=lambda e: e[0])
    row = array.array("i", [0]) * (n + 1)
    col = array.array("i")
    cap = array.array("i")
    for u, v, c in edges:
        row[u + 1] += 1
        col.append(v)
        cap.append(c)
    for u in range(n):
        row[u + 1] += row[u]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, len(edges)))
        for values in (row, col, cap):
            assert values.itemsize == 4
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(f)


def generate(family, n, m, seed, query_count=10, layers=8):
    rng = random.Random(seed)
    if n < 2 * query_count + layers or m < 0:
        raise ValueError("Not enough nodes, or negative edge count")
    queries = [(s, n - query_count + s) for s in range(query_count)]
    edges = []
    if family == "random":
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(1, 10000)) for _ in range(m)]
    elif family in ("layered", "balanced"):
        inner = n - 2 * query_count
        boundaries = [query_count + inner * i // layers for i in range(layers + 1)]
        capacity = lambda: 10000 if family == "balanced" else rng.randint(1, 10000)
        for s in range(query_count):
            edges.extend((s, v, capacity()) for v in range(boundaries[0], boundaries[1]))
        for u in range(boundaries[-2], boundaries[-1]):
            edges.extend((u, t, capacity()) for t in range(n - query_count, n))
        # Backbone guarantees a path through every lane; extra edges allow
        # rerouting across layers. This is a synthetic wide transport network.
        for layer in range(layers - 1):
            next_width = boundaries[layer + 2] - boundaries[layer + 1]
            for u in range(boundaries[layer], boundaries[layer + 1]):
                v = boundaries[layer + 1] + (u - boundaries[layer]) % next_width
                edges.append((u, v, capacity()))
        if len(edges) > m:
            raise ValueError(f"Need at least {len(edges)} edges for these layered dimensions")
        while len(edges) < m:
            layer = rng.randrange(layers - 1)
            edges.append((rng.randrange(boundaries[layer], boundaries[layer + 1]),
                          rng.randrange(boundaries[layer + 1], boundaries[layer + 2]), capacity()))
    else:
        raise ValueError(f"Unknown family: {family}")
    return edges, queries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/layered"))
    p.add_argument("--family", choices=("random", "layered", "balanced"), default="layered")
    p.add_argument("--nodes", type=int, default=100020)
    p.add_argument("--edges", type=int, default=1000000)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    edges, queries = generate(a.family, a.nodes, a.edges, a.seed, a.queries, a.layers)
    write_csr(a.output.with_suffix(".bin"), a.nodes, edges)
    a.output.with_suffix(".queries").write_text("".join(f"{s} {t}\n" for s, t in queries))
    a.output.with_suffix(".json").write_text(json.dumps({
        "family": a.family, "nodes": a.nodes, "edges": len(edges), "seed": a.seed,
        "queries": a.queries, "layers": a.layers}, indent=2) + "\n")
    print(f"Generated {a.output}: N={a.nodes}, M={len(edges)}, Q={len(queries)}")


if __name__ == "__main__":
    main()
