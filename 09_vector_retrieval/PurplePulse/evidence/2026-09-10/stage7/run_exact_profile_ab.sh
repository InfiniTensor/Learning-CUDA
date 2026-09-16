#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-/data/purplepulse-work/PurplePulse}
baseline_binary=${2:-/data/purplepulse-work/jobs/stage7-cosine-norm-build-20260910/vector_search_baseline}
job_dir=${3:-/data/purplepulse-work/jobs/stage7-cosine-norm-ab-20260910}
results_root="$project_root/results/stage7_exact_profile/cosine_norm_ab"

cd "$project_root"
mkdir -p \
  "$results_root/synthetic1m_baseline/outputs" \
  "$results_root/synthetic1m_optimized/outputs" \
  "$results_root/yahoo_baseline/outputs" \
  "$results_root/yahoo_optimized/outputs"

synthetic_configs=10:configs/exact_million_k10_fused_batch64.conf,50:configs/exact_million_k50_fused_batch64.conf,100:configs/exact_million_k100_fused_batch64.conf
synthetic_references=10:results/new_gpu/exact_k10.txt,50:results/new_gpu/exact_k50.txt,100:results/new_gpu/exact_k100.txt
yahoo_configs=10:configs/real_exact_k10.conf,50:configs/real_exact_k50.conf,100:configs/real_exact_k100.conf
yahoo_references=10:results/real/yahoo-minilm-384/exact_outputs/exact_k10.txt,50:results/real/yahoo-minilm-384/exact_outputs/exact_k50.txt,100:results/real/yahoo-minilm-384/exact_outputs/exact_k100.txt

run_profile() {
  local binary=$1
  local database=$2
  local queries=$3
  local configs=$4
  local references=$5
  local output=$6
  python3 scripts/benchmark_exact_breakdown.py \
    --search-binary "$binary" \
    --database "$database" \
    --queries "$queries" \
    --configs "$configs" \
    --reference-results "$references" \
    --output-dir "$results_root/$output/outputs" \
    --csv "$results_root/$output/breakdown.csv" \
    --warmup 5 --repeat 20 --profile-repeat 20
}

run_profile "$baseline_binary" \
  data/million/inner_fp32_db.bin data/million/inner_fp32_q.bin \
  "$synthetic_configs" "$synthetic_references" synthetic1m_baseline
run_profile ./build/vector_search \
  data/million/inner_fp32_db.bin data/million/inner_fp32_q.bin \
  "$synthetic_configs" "$synthetic_references" synthetic1m_optimized
run_profile "$baseline_binary" \
  data/real/yahoo-minilm-384-normalized/database.bin \
  data/real/yahoo-minilm-384-normalized/queries.bin \
  "$yahoo_configs" "$yahoo_references" yahoo_baseline
run_profile ./build/vector_search \
  data/real/yahoo-minilm-384-normalized/database.bin \
  data/real/yahoo-minilm-384-normalized/queries.bin \
  "$yahoo_configs" "$yahoo_references" yahoo_optimized

touch "$job_dir/complete"
