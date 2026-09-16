#!/usr/bin/env python3
"""计算近似 Top-K 相对 exact ground truth 的 recall@K 和分数误差。"""

import argparse
from collections import defaultdict
from pathlib import Path


def read_results(path):
    results = defaultdict(list)
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} 不是三列")
        results[int(parts[0])].append((int(parts[1]), float(parts[2])))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", required=True)
    parser.add_argument("--approximate", required=True)
    args = parser.parse_args()

    exact = read_results(args.exact)
    approximate = read_results(args.approximate)
    if exact.keys() != approximate.keys():
        raise SystemExit("失败：query ID 集合不同")

    matches = 0
    total = 0
    per_query = []
    for query_id in exact:
        if len(exact[query_id]) != len(approximate[query_id]):
            raise SystemExit(f"失败：query {query_id} 的 K 不同")
        exact_ids = {vector_id for vector_id, _ in exact[query_id]}
        approximate_ids = {vector_id for vector_id, _ in approximate[query_id]}
        query_matches = len(exact_ids & approximate_ids)
        matches += query_matches
        total += len(exact[query_id])
        per_query.append(query_matches / len(exact[query_id]))

    score_errors = [
        abs(exact_neighbor[1] - approximate_neighbor[1])
        for query_id in exact
        for exact_neighbor, approximate_neighbor in zip(
            exact[query_id], approximate[query_id]
        )
    ]

    print(
        f"recall@K={matches / total:.6f}, "
        f"min_query_recall={min(per_query):.6f}, "
        f"mean_absolute_score_error={sum(score_errors) / len(score_errors):.9g}, "
        f"max_absolute_score_error={max(score_errors):.9g}, "
        f"queries={len(per_query)}, K={len(next(iter(exact.values())))}"
    )


if __name__ == "__main__":
    main()
