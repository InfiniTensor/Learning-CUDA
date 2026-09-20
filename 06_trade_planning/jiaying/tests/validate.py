#!/usr/bin/env python3
"""Independent Edmonds-Karp oracle, min-cut certificate, and input checks."""
import argparse
import collections
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import write_csr


def oracle(n, edges, source, target):
    residual = [collections.defaultdict(int) for _ in range(n)]
    for u, v, c in edges:
        if u != v:
            residual[u][v] += c
            residual[v][u] += 0
    flow = 0
    while True:
        parent = [-1] * n
        parent[source] = source
        queue = collections.deque([source])
        while queue and parent[target] < 0:
            u = queue.popleft()
            for v, c in residual[u].items():
                if c > 0 and parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
        if parent[target] < 0:
            return flow
        amount, v = 2**63 - 1, target
        while v != source:
            u = parent[v]
            amount = min(amount, residual[u][v])
            v = u
        v = target
        while v != source:
            u = parent[v]
            residual[u][v] -= amount
            residual[v][u] += amount
            v = u
        flow += amount


def run(exe, directory, name, n, edges, queries, expected=None, modes=None):
    graph = directory / (name + ".bin")
    query = directory / (name + ".queries")
    write_csr(graph, n, edges)
    query.write_text("".join(f"{s} {t}\n" for s, t in queries))
    if expected is None:
        expected = [oracle(n, edges, s, t) for s, t in queries]
    if modes is None:
        modes = [("cpu", []), ("cpu-pr", []), ("cuda", ["--check"]),
                 ("cuda", ["--check", "--schedule", "warp", "--no-graphs", "--batch-rounds", "2", "--global-interval", "8"])]
    for index, (engine, options) in enumerate(modes):
        result = directory / f"{name}_{index}.txt"
        log = directory / f"{name}_{index}.json"
        subprocess.run([str(exe), "--graph", str(graph), "--queries", str(query),
                        "--output", str(result), "--log", str(log), "--engine", engine] + options,
                       check=True, capture_output=True, text=True, timeout=90)
        actual = [int(line.split()[2]) for line in result.read_text().splitlines()]
        assert actual == expected, (name, engine, actual, expected)
        data = json.loads(log.read_text())
        assert data["query_count"] == len(queries)
        assert abs(sum(q["ms"] for q in data["queries"]) - data["total_query_ms"]) < 0.001
    return {"name": name, "nodes": n, "edges": len(edges), "queries": len(queries), "modes": len(modes), "passed": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.exe = a.exe.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = []
    textbook = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12),
                (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    queries = [(s, t) for s in range(6) for t in range(6) if s != t]
    cases.append(run(a.exe, a.output, "textbook", 6, textbook, queries + [(0, 5)] * 3))
    assert oracle(6, textbook, 0, 5) == 23
    cases.append(run(a.exe, a.output, "empty", 4, [], [(0, 3), (3, 0)]))
    tricky = [(0, 0, 2147483647), (0, 1, 0), (0, 1, 17), (0, 1, 23), (1, 0, 11),
              (1, 2, 2), (1, 3, 5), (2, 1, 100), (2, 2, 8), (3, 4, 4), (4, 3, 20)]
    cases.append(run(a.exe, a.output, "parallel_cycles_deadends", 8, tricky,
                     [(0, 4), (4, 0), (0, 7), (7, 6), (1, 2), (0, 4)]))
    # Every edge capacity <= 10000, but total flow exceeds signed int32.
    width = 214750
    wide = [(0, 1, 10000)] * width
    cases.append(run(a.exe, a.output, "int64_flow", 3, wide,
                     [(0, 1), (1, 0), (0, 2), (0, 1)],
                     expected=[width * 10000, 0, 0, width * 10000]))
    maximum = [(0, 1, 2147483647), (0, 1, 2147483647), (1, 2, 2147483647), (1, 2, 2147483647)]
    cases.append(run(a.exe, a.output, "int32_edge_boundary", 3, maximum, [(0, 2), (2, 0)]))
    # Disjoint random components allow hundreds of independent oracle checks
    # in one process and exercise repeated-query residual reset.
    rng = random.Random(2039)
    edges, queries, expected = [], [], []
    offset = 0
    for _ in range(100):
        n = rng.randint(2, 35)
        local = [(rng.randrange(n), rng.randrange(n), rng.choice([0, 1, 2, 9999, 10000, rng.randint(1, 10000)]))
                 for _ in range(rng.randint(0, n * 8))]
        edges.extend((u + offset, v + offset, c) for u, v, c in local)
        for _ in range(4):
            s, t = rng.sample(range(n), 2)
            queries.append((s + offset, t + offset))
            expected.append(oracle(n, local, s, t))
        offset += n
    cases.append(run(a.exe, a.output, "random_100_graphs", offset, edges, queries, expected))
    n = 1024
    cases.append(run(a.exe, a.output, "long_chain", n, [(i, i + 1, 10000) for i in range(n - 1)],
                     [(0, n - 1), (n - 1, 0)], expected=[10000, 0],
                     modes=[("cpu", []), ("cpu-pr", []), ("cuda", ["--check", "--global-interval", "2048"])]))
    # Malformed inputs must fail before output creation or CUDA allocation.
    valid = a.output / "invalid_base.bin"
    write_csr(valid, 3, [(0, 1, 3)])
    base = valid.read_bytes()
    bad_graphs = {"short": base[:-1], "trailing": base + b"x", "negative_n": struct.pack("<i", -1) + base[4:],
                  "bad_row": base[:8] + struct.pack("<i", 1) + base[12:],
                  "bad_col": base[:24] + struct.pack("<i", 3) + base[28:],
                  "negative_capacity": base[:-4] + struct.pack("<i", -1),
                  "oversize": struct.pack("<ii", 3, 2147483647)}
    rejected = []
    for name, content in bad_graphs.items():
        path = a.output / f"invalid_{name}.bin"
        path.write_bytes(content)
        query = a.output / "bad.queries"
        query.write_text("0 2\n")
        command = [str(a.exe), "--graph", str(path), "--queries", str(query), "--output",
                   str(a.output / "invalid_out.txt"), "--log", str(a.output / "invalid_log.json")]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert proc.returncode != 0 and "error:" in proc.stderr, name
        rejected.append(name)
    for i, text in enumerate(["0 0\n", "-1 2\n", "0 3\n", "0 2 extra\n", "x y\n", "\n", "2147483648 1\n"]):
        query.write_text(text)
        command[2] = str(valid)
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert proc.returncode != 0, text
        rejected.append(f"bad_query_{i}")
    summary = {"passed": True, "cases": cases, "query_mode_checks": sum(c["queries"] * c["modes"] for c in cases),
               "invalid_inputs_rejected": rejected, "oracle": "independent Python Edmonds-Karp"}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
