#!/usr/bin/env bash
# 性能/回归扫描；参考库严格验收使用 validate_tc.py。
# 结果追加到一个 CSV，便于和 9.1/9.2 的 advanced_results.csv 并列分析。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="${HADAMARD_TC_BIN:-$PROJECT_DIR/build/hadamard_tc_bench}"
OUT_DIR="${HADAMARD_TC_OUT:-$PROJECT_DIR/results}"
CSV="$OUT_DIR/tc_results.csv"

mkdir -p "$OUT_DIR"
if [[ -e "$CSV" ]]; then
    echo "[error] refusing to overwrite $CSV; choose a new HADAMARD_TC_OUT" >&2
    exit 2
fi

if [[ ! -x "$BIN" ]]; then
    echo "[error] benchmark binary not found: $BIN" >&2
    exit 1
fi

status=0

# 主性能矩阵：131072 tokens，与主项目 9.1/9.2 表格同规模，便于横向比较。
for dtype in fp16 bf16; do
    for head_dim in 64 128 256 512 1024; do
        echo "=== $dtype head_dim=$head_dim tokens=131072 ==="
        "$BIN" --batch 4 --seq 1024 --heads 32 --head_dim "$head_dim" \
               --dtype "$dtype" --normalize true --warmup 20 --iters 100 \
               --ref_tokens 4096 --csv "$CSV" || status=1
    done
done

# 语义覆盖：normalize=false 与非 2 的幂倍数 token 数（触发 tail fallback 路径）。
for dtype in fp16 bf16; do
  for head_dim in 64 128 256 512 1024; do
    echo "=== $dtype head_dim=$head_dim normalize=false ==="
    "$BIN" --batch 1 --seq 256 --heads 8 --head_dim "$head_dim" \
           --dtype "$dtype" --normalize false --warmup 5 --iters 20 \
           --ref_tokens 2048 --csv "$CSV" || status=1
  done
done

echo "=== tail fallback: total_tokens 不是每 tile token 数的整数倍 ==="
for dtype in fp16 bf16; do
  for head_dim in 64 128; do
    for tokens in 1 5 1022; do
      "$BIN" --batch 1 --seq 1 --heads "$tokens" --head_dim "$head_dim" --dtype "$dtype" \
             --normalize true --warmup 5 --iters 20 --ref_tokens "$tokens" \
             --csv "$CSV" || status=1
    done
  done
done

echo "[run_tc] csv: $CSV (exit=$status)"
exit $status
