#!/usr/bin/env bash
# run_tests.sh — Week1/Week2 一键批量测试（正确性 + 性能）
#
# 测试矩阵（18 个核心配置 + 4 个分布扩展配置）：
#   dtype    : fp16, bf16
#   head_dim : 64, 128, 256
#   规模     : small(1x128x8=1024 tokens) / medium(2x512x16=16384) / large(4x1024x32=131072)
#   分布     : normal（核心）+ uniform/outlier（扩展）
#
# 输出：
#   results/results.csv   每个配置一行（性能 + 误差）
#
# 用法：
#   bash scripts/run_tests.sh          # 全量
#   bash scripts/run_tests.sh small    # 只跑小规模（冒烟）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/hadamard_bench"
RESULTS="$ROOT/results"
CSV="$RESULTS/results.csv"
WARMUP=5
ITERS=20

if [[ ! -x "$BIN" ]]; then
  echo "[run_tests] 未找到 $BIN，先执行 make ..."
  make -C "$ROOT"
fi

mkdir -p "$RESULTS"
rm -f "$CSV"

# run_config <batch> <seq> <heads> <head_dim> <dtype> [dist]
run_config() {
  local batch=$1 seq=$2 heads=$3 head_dim=$4 dtype=$5 dist=${6:-normal}
  echo ""
  echo "===== dtype=$dtype head_dim=$head_dim batch=$batch seq=$seq heads=$heads dist=$dist ====="
  # bench 在 check 未通过时返回 1：扫描需要继续收集，不能因单配置失败而中断
  if ! "$BIN" \
    --batch "$batch" --seq "$seq" --heads "$heads" --head_dim "$head_dim" \
    --dtype "$dtype" --dist "$dist" --normalize true \
    --warmup "$WARMUP" --iters "$ITERS" --check true --csv "$CSV"; then
    echo "[run_tests] ^^^ 该配置 check 未通过（详见 CSV/报告，继续扫描）"
  fi
}

MODE=${1:-all}

if [[ "$MODE" == "small" || "$MODE" == "all" ]]; then
  for dtype in fp16 bf16; do
    for hd in 64 128 256; do
      run_config 1 128 8 "$hd" "$dtype"
    done
  done
fi

if [[ "$MODE" == "all" ]]; then
  for dtype in fp16 bf16; do
    for hd in 64 128 256; do
      run_config 2 512 16 "$hd" "$dtype"
      run_config 4 1024 32 "$hd" "$dtype"
    done
  done
  # 分布扩展（中等规模）
  for dtype in fp16 bf16; do
    run_config 2 512 16 128 "$dtype" uniform
    run_config 2 512 16 128 "$dtype" outlier
  done
fi

# 官方库验收测试：与 fast_hadamard_transform 逐位对照（项目验收标准）
# 需要 ~/hadamard_env（torch + fast_hadamard_transform），搭建方法见 README
if [[ -x "$HOME/hadamard_env/bin/python" ]]; then
  echo ""
  echo "[run_tests] 官方库验收测试 (vs fast_hadamard_transform) ..."
  if ! "$HOME/hadamard_env/bin/python" "$ROOT/tests/test_vs_library.py"; then
    echo "[run_tests] 官方库验收存在未通过配置，详见上方输出"
  fi
else
  echo ""
  echo "[run_tests] 未发现 ~/hadamard_env，跳过官方库验收（搭建方法见 README '依赖安装'）"
fi

# 汇总通过情况（最后一列 pass）
TOTAL=$(( $(wc -l < "$CSV") - 1 ))
PASS=$(awk -F, 'NR>1 && $NF=="true"' "$CSV" | wc -l)
echo "[run_tests] 完成：$TOTAL 个配置，$PASS 个通过；结果在 $CSV"
if [[ "$PASS" -ne "$TOTAL" ]]; then
  echo "[run_tests] 注意：存在未通过绝对阈值的配置；BF16 outlier 的舍入边界见 docs/final_report.md"
fi
