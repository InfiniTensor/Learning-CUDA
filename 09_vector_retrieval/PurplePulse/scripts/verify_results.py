#!/usr/bin/env python3
"""比较 CPU 与 GPU 的文本结果，检查 Top-K ID 和分数误差。"""

import argparse
from collections import defaultdict
from pathlib import Path


def read_results(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number} 不是三列")
        rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-5,
        help="CPU 串行累加与 GPU warp 归约的允许绝对误差",
    )
    args = parser.parse_args()

    expected = read_results(args.expected)
    actual = read_results(args.actual)
    if len(expected) != len(actual):
        raise SystemExit(
            f"失败：行数不同，expected={len(expected)}, actual={len(actual)}"
        )

    expected_by_query = defaultdict(list)
    actual_by_query = defaultdict(list)
    for row in expected:
        expected_by_query[row[0]].append(row)
    for row in actual:
        actual_by_query[row[0]].append(row)

    if expected_by_query.keys() != actual_by_query.keys():
        raise SystemExit("失败：query ID 集合不同")

    max_error = 0.0
    error_sum = 0.0
    error_count = 0
    near_tie_reorders = 0
    for query_id in expected_by_query:
        expected_rows = expected_by_query[query_id]
        actual_rows = actual_by_query[query_id]
        expected_scores = {row[1]: row[2] for row in expected_rows}
        actual_scores = {row[1]: row[2] for row in actual_rows}
        if expected_scores.keys() != actual_scores.keys():
            missing = expected_scores.keys() - actual_scores.keys()
            unexpected = actual_scores.keys() - expected_scores.keys()
            raise SystemExit(
                f"失败：query {query_id} 的 Top-K 集合不同，"
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for vector_id in expected_scores:
            error = abs(expected_scores[vector_id] - actual_scores[vector_id])
            max_error = max(max_error, error)
            error_sum += error
            error_count += 1
        for rank, (left, right) in enumerate(zip(expected_rows, actual_rows)):
            if left[1] != right[1]:
                score_gap = abs(left[2] - right[2])
                if score_gap > args.tolerance:
                    raise SystemExit(
                        f"失败：query {query_id} rank {rank} 顺序不同且非近似并列，"
                        f"expected={left}, actual={right}"
                    )
                near_tie_reorders += 1

    if max_error > args.tolerance:
        raise SystemExit(
            f"失败：最大分数误差 {max_error} 超过容差 {args.tolerance}"
        )
    print(
        "通过：Top-K 候选集合相同，"
        f"平均绝对分数误差={error_sum / error_count if error_count else 0.0}，"
        f"最大分数误差={max_error}，近似并列换位数={near_tie_reorders}"
    )


if __name__ == "__main__":
    main()
